import os
from googlesearch import search
import unicodedata
import io
import Levenshtein
import urllib.request
from urllib.error import HTTPError
import re
import time
import string
import sys
from bs4 import BeautifulSoup
from unidecode import unidecode
import inflect
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import bisect
import math
from difflib import SequenceMatcher
import os
import wget
import tarfile
import argparse

html_pool = "./TEDLIUM_dataset/TEDLIUM_release2/tedlium_html_pool"
error_log_dir = "./TEDLIUM_dataset/TEDLIUM_release2/tedlium_error_tracking"
if not os.path.exists(html_pool):
    os.mkdir(html_pool)
if not os.path.exists(error_log_dir):
    os.mkdir(error_log_dir)

error_404_link_log = os.path.join(error_log_dir, "error_url_404.log")
error_speaker_404_log = os.path.join(error_log_dir, "error_speaker_404.log")
error_google_speaker_404_log = os.path.join(error_log_dir, "error_google_speaker_404.log")
error_ted_speaker_404_log = os.path.join(error_log_dir, "error_ted_speaker_404.log")
error_no_online_transcript_log = os.path.join(error_log_dir, "error_no_online_transcript_404.log")
error_all_others_log = os.path.join(error_log_dir, "error_all_other.log")
similarity_score_log = os.path.join(error_log_dir, "similarity_score.log")


with open(similarity_score_log, "w") as fh:
    pass

valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)

music_char = "♫"

DEBUG = 0

class CustomWebError404(Exception):
    pass

class CustomWebError429(Exception):
    pass

class CustomWebError503(Exception):
    pass

class CustomWebErrorGoogle404(Exception):
    def __init__(self, message):
        super(CustomWebErrorGoogle404, self).__init__(message)

class CustomWebErrorNoOnlineTranscript(Exception):
    def __init__(self, message):
        super(CustomWebErrorNoOnlineTranscript, self).__init__(message)

class GavinNoPuncForSTM(Exception):
    pass

def write_unique_msg_to_file(filepath, msg):
    """
    Write msg to filepath (add newline if not exist).
    Create file if filepath does not exist.
    """
    msg = msg.rstrip("\n")
    early_exit = False
    if not os.path.exists(filepath):
        with open(filepath, "w") as fh:
            pass
    with open(filepath, "r+") as fh:
        for line in fh:
            if msg == line.rstrip("\n"):
                early_exit = True
                break
            else:
                fh.write(msg + "\n")
        if not early_exit:
            fh.write(msg + "\n")


def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file: the stm file processed
    :return: a single string containing the entire transcript with all other information removed
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            if stm_line.rstrip("\n").lstrip("\n"):
                tokens = stm_line.split()
                filename = tokens[0]

                token_1 = tokens[1]
                token_2 = tokens[2]

                start_time = float(tokens[3])
                end_time = float(tokens[4])

                token_5 = tokens[5]

                transcript = unicodedata.normalize("NFKD",
                                                   " ".join(t for t in tokens[6:]).strip()). \
                    encode("utf-8", "ignore").decode("utf-8", "ignore")
                if transcript != "ignore_time_segment_in_scoring":
                    res.append({
                        "start_time": start_time, "end_time": end_time,
                        "filename": filename, "transcript": transcript,
                        "token_1": token_1, "token_2": token_2,
                        "token_5": token_5
                    })
        return res


def write_stm_from_utterances(stm_file, res_list):
    with io.open(stm_file, "w", encoding='utf-8') as f:
        line_list = []
        for res in res_list:
            # token 0
            filename = str(res['filename'])

            # token 1 and 2
            token_1 = str(res['token_1'])
            token_2 = str(res['token_2'])

            # token 3 and 4
            start_time = str(res['start_time'])
            end_time = str(res['end_time'])

            # token 5
            token_5 = str(res['token_5'])

            # token 6+
            transcript = str(res['transcript'])

            line_list.append(" ".join([filename, token_1, token_2, start_time, end_time, token_5, transcript]) + "\n")

        f.writelines(line_list)


def convert_string_to_filename(link):
    """
    Convert a html string to a valid filename string.
    This function is designed to record an html link as a filename. However, html link contains characters
    not valid as part of the filename. Therefore, these characters need to be converted to their unicode equivalent,
    marked by surrounding X.
    """
    filename_tokens = []
    for char in link:
        if char in valid_chars:
            filename_tokens.append(char)
        else:
            filename_tokens.append("_X" + str(ord(char)) + "X_")
    filename = ''.join(filename_tokens)
    if len(filename) > 250:
        filename = filename[0:250]
    return filename


def convert_filename_to_string(filename):
    """
    Reverse function of convert_string_to_filename.
    """
    link = re.sub(r'_X(\d+)X_', lambda x: chr(int(x.group(1))), filename)
    return link


def write_unique_msg_to_local_link(link, msg):
    link_filename = convert_string_to_filename(link)
    write_unique_msg_to_file(os.path.join(html_pool, link_filename), msg)


def read_html_from_link_pool(link, html_path=None):
    """
    Read html from html link pool if exists. Otherwise, read by querying the internet.
    :param html_path: Extra location to save the html corresponding to link.
    :return: The html string corresponding to link
    """
    link_filename = convert_string_to_filename(link)
    pool_filepath = os.path.join(html_pool, link_filename)
    if os.path.exists(pool_filepath):
        with open(pool_filepath) as fh:
            html_str = fh.read()
            #print("[HTML Local] Reading Link: {}".format(link))
    else:
        print("[Networking] Reading Link: {}".format(link))
        try:
            html_fh = urllib.request.urlopen(link)
        except HTTPError as e:
            if e.code == 404:
                write_unique_msg_to_file(error_404_link_log, link)
                raise CustomWebError404
            else:
                raise e
        html_str = html_fh.read().decode("utf8")
        with open(pool_filepath, "w") as fh:
            fh.write(html_str)
        if html_path:
            with open(html_path, "w") as fh:
                fh.write(html_str)
    return html_str


def read_html_from_link(link, html_path=None):
    """
    Read html from link. Retry after backoff seconds if error 429 (we are making too frequent requests) is received.
    :param html_path: Extra location to save the html
    :return: the html in string
    """
    backoff = 35
    while True:
        try: # Retry if error 429 is received
            # time information if pausing for error 429 is required
            time_diff = time.time() - read_html_from_link.timestamp
            if time_diff > 30:
                print("Seconds since last retry: {}".format(time_diff))
            read_html_from_link.timestamp = time.time()

            html_str = read_html_from_link_pool(link, html_path)
            return html_str
        except HTTPError as e:
            if e.code == 429:
                time.sleep(backoff)
                backoff = backoff**1.5
            else:
                raise e
read_html_from_link.timestamp = time.time()


def read_soup_from_link(link):
    """
    Convert link to beautiful soup dictionary representation.
    :return: BS object.
    """
    html_str = read_html_from_link(link)
    soup = BeautifulSoup(html_str, "html.parser")
    return soup


def get_en_links(all_links):
    """
    Get all english transcript links only.
    """
    truncated_links = set()
    for link in all_links:
        truncated_link = re.search(".*transcript", link)
        truncated_links.add(truncated_link.group(0))
    return list(truncated_links)


def get_talk_links(speaker_link):
    """
    Get the talk links on a ted speaker page.
    :return: a list of all the talk links on the speaker's page.
    """
    talk_links = []
    soup = read_soup_from_link(speaker_link)
    tag_a = soup.find_all('a')
    for tag in tag_a:
        href = tag.attrs.get('href')
        if href and href.startswith("/talks/"):
            talk_link = "https://www.ted.com" + href + "/transcript"
            talk_links.append(talk_link)
    return talk_links


def get_manual_result_link(msg):
    """
    Ask the user to input an link.
    :param msg: the msg prompt
    :return: the link that user provided
    """
    one_link = ""
    while not one_link:
        if one_link == "break":
            break
        one_link = input(msg)
    if one_link == "break":
        one_link = ""
    return one_link


def read_result_from_search_pool(google_search_domain, google_search_speaker):
    """
    Get the search result speaker link corresponding to a google search term.
    :param goo_search_term: the google search term (should include site: if needed)
    :return: a list of speaker links.
    """
    goo_search_term = google_search_domain + google_search_speaker

    # get the corresponding file that is storing the search result
    search_filename = convert_string_to_filename(goo_search_term)
    pool_filepath = os.path.join(html_pool, search_filename)

    # use the content in the file if it exists
    if os.path.exists(pool_filepath):
        with open(pool_filepath) as fh:
            search_result_links = fh.readlines()
            if search_result_links:
                return search_result_links
    # perform actual google search if it does not exist
    print("[Networking] Searching Google: {}".format(goo_search_term))
    search_results = None
    one_link = ""
    try:
        search_results = search(goo_search_term, num=5)
        search_results = list(search_results)
        if not search_results:
            search_results = search("ted speaker " + google_search_speaker, num=5)
            search_results = list(search_results)
            if not search_results:
                # one_link = get_manual_result_link("[Google No Result] Google Search of < {} > Failed,\nYou Can Enter a Speaker Link Manually: ".format(goo_search_term))
                raise CustomWebErrorGoogle404("[Google Search Failed]" + goo_search_term)
            else:
                search_results = [result for result in search_results if "https://www.ted.com/speakers" in result]
    except HTTPError as e:
        if e.code == 503:
            one_link = get_manual_result_link("[Too Many 503] Google Search of < {} > Failed,\nYou Can Enter a Speaker Link Manually: ".format(goo_search_term))
        else:
            raise e

    if search_results:
        # remove the page with list of speakers, this is not what we are looking for
        search_results = [search_result for search_result in search_results if ("?" not in search_result) and ("=" not in search_result)]
        search_results = [search_result for search_result in search_results if 'https://www.ted.com/speakers/' in search_result]
        speaker_names = [search_result.replace('https://www.ted.com/speakers/', '') for search_result in search_results]
        if not speaker_names:
            raise CustomWebErrorGoogle404("[Google Search Failed]" + goo_search_term)
        speaker_similarity = [SequenceMatcher(None, speaker_name, google_search_speaker).ratio() for speaker_name in speaker_names]
        speaker_index = speaker_similarity.index(max(speaker_similarity))
        speaker_link = search_results[speaker_index]
    elif one_link:
        write_unique_msg_to_local_link(goo_search_term, one_link)
        speaker_link = [one_link]
    else:
        raise CustomWebErrorNoOnlineTranscript(google_search_speaker)
    with open(pool_filepath, "w") as fh:
        fh.writelines([speaker_link + "\n"])
    search_result_links = []
    search_result_links.append(speaker_link)
    return search_result_links


def get_search_name_from_speaker_name(speaker_name):
    """
    Get both the google search name, and best guess of the netfix domain name
    :param speaker_name: Speaker name in its raw format (as appeared in the filename)
    :return: (google search name, ted_speaker_name)
    """
    temp_list = speaker_name.split("_")
    name = temp_list[0]
    camelName = re.sub("[A-Z]{2,}", lambda x: x.group(0).lower().capitalize(), name)
    speaker_search_name = ''.join(map(lambda x: " " + x.lower() if x.isupper() else x, camelName)).rstrip(" ").lstrip(" ")
    ted_speaker_name = ''.join(map(lambda x: "_" + x.lower() if x.isupper() else x, camelName)).rstrip("_").lstrip("_")
    return speaker_search_name, ted_speaker_name


def read_transcript_from_link(link):
    """
    Read transcript of a given link. The link should correspond to a ted.com/talks/*/transcript page
    :return: The transcript in a string.
    """
    soup = read_soup_from_link(link)
    sentence_list = []
    for tag in soup.find_all('p'):
        sentence = re.sub(r"\s+", " ", tag.text)
        sentence_list.append(sentence)
    transcript = " ".join(sentence_list)
    return transcript


def calculate_similarity_score(plain_file, links):
    """
    Given a list of links, calcualte the similarity score between the transcripts in the links and the plain_file.
    :return: The similarity score in a list.
    """
    similarity_list = []
    for link in links:
        try:
            online_transcript = read_transcript_from_link(link)
        except CustomWebError404:
            online_transcript = ""
        with open(plain_file) as fh:
            # clean online transcript
            online_to_comp = generate_clean_punc(online_transcript, for_comp=True)

            # clean stm transcript
            plain_file_list = fh.readlines()
            stm_transcript = " ".join(plain_file_list)
            stm_to_comp = stm_transcript.replace("<unk>", " ").replace(" 's", "'s").replace(" 'd", "'d").replace(" 't", "'t").replace(" 're", "'re").replace(" 've", "'ve").replace(" 'll", "'ll").replace(" 'm", "'m").replace("  ", " ")

            similarity = Levenshtein.ratio(online_to_comp, stm_to_comp)
            if similarity == 0.0:
                dummy = 1
            similarity_list.append(similarity)
    return similarity_list


def write_html_with_name(html_dir, plain_file, speaker_name):

    google_speaker_search_name, ted_speaker_name = get_search_name_from_speaker_name(speaker_name)

    ted_speaker_base = "https://www.ted.com/speakers/"
    ted_speaker_link = ted_speaker_base + ted_speaker_name

    # try get speaker talk links by first going through TED, then go through google search results
    try:
        talk_links = get_talk_links(ted_speaker_link)
    except CustomWebError404 as e: # ted speaker link invalid, retry by performing google search of speaker
        # Write the speaker to the list of speaker that does not have successful ted information
        #print("Speaker suffered ted 404 error: {}".format(speaker_name))
        write_unique_msg_to_file(error_ted_speaker_404_log, speaker_name)

        google_search_domain = "site:https://www.ted.com/speakers/ "
        search_result_links = read_result_from_search_pool(google_search_domain, google_speaker_search_name)

        # get a union of talk links that are on any of the result link
        talk_links = set()
        for link in search_result_links:
            talk_links.update(get_talk_links(link))

    # Get the link with english transcript for speaker
    # Exit the function and log the information if no english transcript found
    eng_links = get_en_links(talk_links)
    if not eng_links:
        print("[All Attempt Failed] NO LINKS FOUND for Speaker: {}".format(speaker_name))
        for link in talk_links:
            print(link)
        write_unique_msg_to_file(error_speaker_404_log, speaker_name)
        return None

    similarity_list = calculate_similarity_score(plain_file, eng_links)
    if True:
        with open(similarity_score_log, "a") as fh:
            print("")
            fh.write("\n")
            temp = "{:100s} {:3.3f}".format(ted_speaker_link, max(similarity_list))
            print(temp)
            fh.write(temp + "\n")
            for link, similarity in zip(eng_links, similarity_list):
                temp = "{:100s} {:3.3f}".format(link, similarity)
                print(temp)
                fh.write(temp + "\n")

    if not similarity_list:
        raise CustomWebErrorNoOnlineTranscript("No Transcript Found")
    else:
        max_index = similarity_list.index(max(similarity_list))
        best_html = read_html_from_link(eng_links[max_index], os.path.join(html_dir, speaker_name + ".html"))
        if max(similarity_list) > 0.1:
            return eng_links[max_index]
        else:
            raise CustomWebErrorNoOnlineTranscript(eng_links[max_index] + " " + str(max(similarity_list)))


def write_raw_punc_with_name(rawpunc_dir, best_link, speaker_name):
    transcript = read_transcript_from_link(best_link)
    with open(os.path.join(rawpunc_dir, speaker_name + ".txt"), "w") as fh:
        fh.write(transcript)
    return transcript

def clean_non_utf8(transcript):
    """
    Remove all non UTF 8 Characters from the transcript.
    :param transcript: the input transcript
    :return: transcript with non UTF 8 characters removed
    """
    charset = set(transcript)
    for c in charset:
        if c != unidecode(c) and len(unidecode(c)) != 1:
            transcript.replace(c, "")
    transcript = unidecode(transcript)
    return transcript


def select_high_priority_punctuation(punc_string):
    modified = punc_string.replace(" ", "")
    if modified == "'":
        return punc_string
    if not modified:
        print("[Punctuation Selection Warning] This should not occur, punc_string is: {}".format(punc_string))
        return " "
    priority = "?!.,"
    punc_list = list(set(modified))
    if len(punc_list) > 1:
        for known_punc in priority:
            if known_punc in punc_list:
                return known_punc
        return modified[0]
    else:
        return punc_list[0]


def remove_connected_string(transcript):
    transcript = re.sub("([^\w\s]+\s*)+", lambda x: select_high_priority_punctuation(x.group(0)), transcript)
    return transcript


def convert_number_to_string(transcript):
    p = inflect.engine()

    # Convert dollar amount, TODO: reading of money can be very different, this is an approximation that works in many cases
    transcript = re.sub("\$\d*([,\.]?\d+)+", lambda x: p.number_to_words(x.group(0).replace("$", "")) + " dollar ", transcript)
    # Convert float and number with comma
    transcript = re.sub("\d+([,\.]\d+)+", lambda x: p.number_to_words(x.group(0)).replace(",", ""), transcript)
    # Convert number
    transcript = re.sub("\d+", lambda x: p.number_to_words(x.group(0)).replace(",", ""), transcript)
    return transcript


def generate_clean_punc(transcript, for_comp=False, remove_trailing_space=False):

    if not transcript:
        return transcript

    clean_transcript = transcript.replace("©", " ")

    # Convert bullet points marked by a), b) ...
    clean_transcript = re.sub("\s([A-Fa-f])\)", lambda x: " " + x.group(1) + " ", clean_transcript)

    # Remove unnecessary colon that is used to inidicate which speaker is speaking
    clean_transcript = re.sub("([.?!])(\s*[A-Z]\w+|\s*#\d){1,}:", lambda x: " " + x.group(1) + " ", clean_transcript)

    # Remove \
    clean_transcript = transcript.replace("\\", " ")

    # Replace semicolon and colon with period
    clean_transcript = clean_transcript.replace(";", ".").replace(":", ".")

    # Convert # to "number" if followed by a number, otherwise convert it to hashtag
    clean_transcript = re.sub("#(\d+)", lambda x: "number " + x.group(1), clean_transcript)
    clean_transcript = clean_transcript.replace("#", " hashtag ")

    # Clean St
    clean_transcript = clean_transcript.replace("St.", "St")

    # Convert percentage
    clean_transcript = clean_transcript.replace("%", " percent")

    # Convert number to reading
    clean_transcript = convert_number_to_string(clean_transcript)

    # Replacement of non utf-8 chars
    # Some chars we want to keep
    clean_transcript = clean_transcript.replace("—", ",")
    clean_transcript = clean_transcript.replace("æ", "ae")
    clean_transcript = clean_transcript.replace("…", ".")
    clean_transcript = clean_transcript.replace("²", " square ")
    clean_transcript = clean_transcript.replace("˚", " degree ")

    # Throw away the rest
    clean_transcript = clean_non_utf8(clean_transcript)

    # Replace "-" with " "
    clean_transcript = clean_transcript.replace("-", " ")

    # Convert to unicode representation
    clean_transcript = unidecode(clean_transcript)

    # Convert math operations, TODO: not entirely accurate but mostly correct
    clean_transcript = clean_transcript.replace("*", " times ").replace("+", " plus ")

    # Convert /, TODO: too many meanings, convert to blank
    clean_transcript = clean_transcript.replace("/", " ")

    # Convert &, TODO: mostly correct
    clean_transcript = clean_transcript.replace("&", " and ")

    # Convert =, TODO: may be equal or equals
    clean_transcript = clean_transcript.replace("=", " equal ")

    # Remove ` and _   ( "`" and "_" is usually a result of some non-unicode characters)
    clean_transcript = clean_transcript.replace("`", " ").replace("_", " ")

    # Convert @ to at
    clean_transcript = clean_transcript.replace("@", " at ")

    # Remove ^
    clean_transcript = clean_transcript.replace("^", " ")

    # Remove $, these left over $ signs are unexpected, simply remove them
    clean_transcript = clean_transcript.replace("$", " ")

    # Lower case
    clean_transcript = clean_transcript.lower()

    # Remove everything in round bracket
    round_bracket_pattern = "\(.*?\)"
    clean_transcript = re.sub(round_bracket_pattern, "", clean_transcript)

    # Remove ), TODO: usually due to unable to properly match nested brackets
    clean_transcript = clean_transcript.replace(")", " ")

    # Remove everything in square bracket
    square_bracket_pattern = "\[.*?\]"
    clean_transcript = re.sub(square_bracket_pattern, "", clean_transcript)

    # Remove ' if it is enclosing a word
    single_quote_bracket_pattern = "'(\w+)'"
    clean_transcript = re.sub(single_quote_bracket_pattern, lambda x: x.group(1), clean_transcript)

    # Remove . inside acronyms
    acronym_with_dot_pattern = "(\w\.)+"
    clean_transcript = re.sub(acronym_with_dot_pattern, lambda x: x.group(0).replace(".", " "), clean_transcript)

    # Add space before '
    clean_transcript = re.sub("'", " '", clean_transcript)

    # Replace " with .
    clean_transcript = clean_transcript.replace('"', ",")

    # Give punctuation extra space
    extra_space_punct = set(string.punctuation)
    extra_space_punct.remove("'")
    clean_transcript = ''.join(ch if (ch not in extra_space_punct) else (" " + ch + " ") for ch in clean_transcript)

    # Reduce connected punctuation to only one punctuation
    clean_transcript = remove_connected_string(clean_transcript)

    # Give punctuation extra space again because the previous operation may have messed it up
    clean_transcript = ''.join(ch if (ch not in extra_space_punct) else (" " + ch + " ") for ch in clean_transcript)


    if for_comp:
        clean_transcript = ''.join(ch for ch in clean_transcript if ch not in extra_space_punct)

    # Remove unnecessary space
    clean_transcript = re.sub("\s+", " ", clean_transcript)

    # Remove trademark
    clean_transcript = clean_transcript.replace("ted com translations are made possible by volunteer translators learn more about the open translation project ted conferences , llc all rights reserved", "")
    clean_transcript = clean_transcript.replace("ted conferences , llc all rights reserved", "")

    if remove_trailing_space:
        clean_transcript = clean_transcript.lstrip(" ").rstrip(" ")

    return clean_transcript


def write_clean_punc_with_name(punc_dir, rawpunc_dir, speaker_name):
    punc_file = os.path.join(punc_dir, speaker_name + ".txt")
    rawpunc_file = os.path.join(rawpunc_dir, speaker_name + ".txt")
    # if not "ZainabSalbi" in speaker_name:
    #     return
    # if os.path.exists(punc_file):
    #     return
    if os.path.exists(rawpunc_file):
        with open(rawpunc_file) as fh:
            transcript = fh.read()
            # if "ZainabSalbi" in speaker_name:
            #     dummy = 1
            clean_transcript = generate_clean_punc(transcript)
        with open(punc_file, "w") as fh:
            fh.write(clean_transcript)


def write_puncstm_with_name(punc_dir, stm_dir, puncstm_dir, speaker_name):
    get_plain_with_stm(None, stm_dir, speaker_name, get_list=True)
    pass


def get_plain_with_stm(plain_dir, stm_dir, speaker_name, get_list=False):
    """
    Join all transcript content in stm file together and produce a plain text transcript.
    """
    all_utterances = get_utterances_from_stm(os.path.join(stm_dir, speaker_name + ".stm"))
    plain_txt_list = [utter['transcript'] for utter in all_utterances]
    if get_list:
        return plain_txt_list
    plain_txt = " ".join(plain_txt_list)
    plain_file = os.path.join(plain_dir, speaker_name + ".txt")

    if not os.path.exists(plain_file):
        with open(plain_file, "w") as fh:
            fh.write(plain_txt)

    return plain_file





def create_dir(base_path, subfolder):
    new_dir = os.path.join(base_path, subfolder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir

def process_alignment(alignment):
    alignment_index = []
    for i, (x, y) in enumerate(zip(alignment[0], alignment[1])):
        if x == y:
            alignment_index.append(i)
    transcript_section = alignment[0][min(alignment_index):(max(alignment_index)+1)]
    transcript = transcript_section.replace("-", "")
    return transcript

def find_best_alignment(online_transcript, original_stm_trans):
    correct_bonus = 1
    open_gap_penalty = -5
    if len(original_stm_trans) <= abs(open_gap_penalty):
        correct_bonus = round(correct_bonus * abs(open_gap_penalty) / len(original_stm_trans), 1)
    alignments = pairwise2.align.globalms(online_transcript, original_stm_trans, correct_bonus, -1, open_gap_penalty, -0.01)

    # for alignment in alignments:
    #     print(format_alignment(*alignment))

    best_alignment = None
    best_split_word_list = None
    best_split_number = math.inf

    for alignment in alignments:
        split_word_list = list(re.finditer("\w+-+\w+", alignment[1]))
        if not split_word_list:
            return alignment
        else:
            if best_split_number > len(split_word_list):
                best_split_number = len(split_word_list)
                best_alignment = alignment
                best_split_word_list = split_word_list

    print(format_alignment(*best_alignment))
    print(best_split_word_list)
    best_alignment = list(best_alignment)

    for split_word in best_split_word_list:
        span = split_word.span()
        begin = span[0]
        end = span[1]
        original_word = split_word.group(0)

        dash_seq = re.search("-+", original_word).group(0)

        new_word = original_word.replace("-", "")

        if best_alignment[0][begin:begin + len(new_word)] == new_word:
            new_word = new_word + dash_seq
            best_alignment[1] = best_alignment[1][:begin] + new_word + best_alignment[1][begin+len(new_word):]
            assert(len(best_alignment[0]) == len(best_alignment[1]))

    print(format_alignment(*best_alignment))
    new_split_word_list = list(re.finditer("\w+-+\w+", best_alignment[1]))
    print(new_split_word_list)
    return best_alignment

class AugmentedAlignmentList:
    def __init__(self, online_transcript, stm_line):
        self.alignment_list = []
        self.optimal_alignment = None
        self.online_transcript = online_transcript
        self.stm_line = stm_line

    def add_alignment(self, alignment):
        assert isinstance(alignment, AugmentedAlignment)
        self.alignment_list.append(alignment)

    def get_least_difference(self):
        least_diff_alignment = min(self.alignment_list, key=lambda x: x.max_index - x.min_index)
        least_difference = least_diff_alignment.max_index - least_diff_alignment.min_index
        return least_difference

    def get_okay_alignment(self):
        least_difference = self.get_least_difference()
        okay_alignment = [align for align in self.alignment_list if align.max_index - align.min_index == least_difference]
        return okay_alignment

    def get_optimal_alignment(self, previous_alignment=None):
        if self.optimal_alignment:
            return self.optimal_alignment
        okay_alignment = self.get_okay_alignment()
        if previous_alignment:
            # if given previous_alignment, find the alignment with min_index right after the previous_alignment
            after_alignment = [align for align in okay_alignment if align.min_index > previous_alignment.max_index]
            if after_alignment:
                optimal_alignment = min(after_alignment, key=lambda x: x.min_index)
                self.optimal_alignment = optimal_alignment
                return optimal_alignment
            else: # no after_alignment found, likely due to a common word with the previous script
                optimal_alignment = min(okay_alignment, key=lambda x: x.min_index)
                if optimal_alignment.first_word == previous_alignment.last_word:
                    common_word = optimal_alignment.first_word
                    # Common word found and index difference as expected
                    if optimal_alignment.min_index == previous_alignment + len(common_word):
                        self.optimal_alignment = optimal_alignment
                        optimal_alignment.common_with_previous = common_word
                        previous_alignment.common_with_next = common_word
                        return self.optimal_alignment
                    else:
                        print("[Alignment Error] CurrentSTM: {}, PreviousSTM: {}, Common word is: {}, Index Mismatch, Current Optimal: {}, {}, Previous Optimal: {}, {}".format(optimal_alignment.original_stm, previous_alignment.original_stm, common_word, optimal_alignment.min_index, optimal_alignment.max_index, previous_alignment.min_index, previous_alignment.max_index))
                        dummy = 1
                else: # Common word not found, this is likely a problem where first word in this transcript line is same as the first word in the entire transcript, therefore, an alignment search need to be done again
                    if self.online_transcript[0] == self.stm_line[0]:
                        dummy = 0 #TODO
                    print("[Alignment Error] CurrentSTM: {}, PreviousSTM: {}, No Common word, Index Mismatch, Current Optimal: {}, {}, Previous Optimal: {}, {}".format(optimal_alignment.original_stm, previous_alignment.original_stm, optimal_alignment.min_index, optimal_alignment.max_index, previous_alignment.min_index, previous_alignment.max_index))
                    dummy = 1

        # if previous alignment no given, return the alignment that is the closest to beginning
        optimal_alignment = min(okay_alignment, key=lambda x: x.min_index)
        self.optimal_alignment = optimal_alignment
        return optimal_alignment

class AugmentedAlignment_Original:
    def __init__(self, alignment, original_stm_trans, online_transcript):
        self.alignment = alignment

        self.min_index, self.max_index, self.punc_transcript = self.find_min_max_index(alignment, online_transcript)

        self.original_stm = original_stm_trans

        self.first_word = original_stm_trans.split(" ")[0]
        self.last_word = original_stm_trans.split(" ")[-1]

        self.common_with_previous = None
        self.common_with_next = None

    def find_min_max_index(self, alignment, online_transcript):
        # Find all index in that have alignment[0] and [1] the same
        alignment_index = []
        for i, (x, y) in enumerate(zip(alignment[0], alignment[1])):
            if x == y:
                alignment_index.append(i)
        punc_transcript_with_dash = alignment[0][min(alignment_index):(max(alignment_index) + 1)]

        # Find the index of all dashes
        dash_index = []
        for i, x in enumerate(alignment[0]):
            if x == "-":
                dash_index.append(i)

        # Find the actual index of alignment, after excluding the effect of dashes
        new_alignment_index = []
        for match_idx in alignment_index:
            dash_before_me = bisect.bisect_left(dash_index, match_idx)
            new_alignment_index.append(match_idx - dash_before_me)

        # Find the index of the punc_transcript in the original online_transcript
        min_index = min(new_alignment_index)
        max_index = max(new_alignment_index)

        punc_transcript = punc_transcript_with_dash.replace("-", "")

        return min_index, max_index, punc_transcript

def update_res_list(res_list, punc_txt_path):
    with open(punc_txt_path) as fh:
        online_transcript = fh.read().rstrip("\n").lstrip(" ").rstrip(" ")

        all_res_aa_list = [] # all_res_aa_list contains list of aa_list
        # Iterate through transcript line in stm line by line
        for res_index, res in enumerate(res_list):

            # For Debugging
            if res_index == 6:
                dummy = 1

            # Get and clean original stm transcript
            original_stm_trans = res['transcript']
            original_stm_trans = original_stm_trans.replace("<unk>", "").lstrip(" ").rstrip(" ")

            # Find all alignments
            alignments = find_best_alignment(online_transcript, original_stm_trans)

            aa_list = AugmentedAlignmentList(online_transcript, original_stm_trans) # aa_list contains all alignments for one line of transcript
            for i, a in enumerate(alignments):
                aa = AugmentedAlignment(a, original_stm_trans, online_transcript)
                aa_list.add_alignment(aa)

            # Find the optimal alignment for current line
            if all_res_aa_list:
                optimal_alignment = aa_list.get_optimal_alignment(all_res_aa_list[-1].get_optimal_alignment())
            else:
                optimal_alignment = aa_list.get_optimal_alignment()

            # For Debugging
            print("res_index: {}".format(res_index))
            print(format_alignment(*optimal_alignment.alignment))

            # Add all alignment for current line to all_res_aa_list for future reference
            all_res_aa_list.append(aa_list)

    if len(res_list) != len(all_res_aa_list):
        print("[Alignment Not Valid] res_list len: {}, all_res_aa_list len: {}".format(len(res_list), len(all_res_aa_list)))

    # update res_list for all elements
    for res_index, res in enumerate(res_list):
        aa_list = all_res_aa_list[res_index]
        optimal_alignment = aa_list.get_optimal_alignment()
        res_list[res_index]["transcript"] = optimal_alignment.punc_transcript

    return res_list

class AugmentedAlignment:
    def __init__(self, alignment, online_transcript, stm_transcript_all, stm_transcript_list, res_list):
        stm_transcript_list = stm_transcript_list[:-1]
        res_list = res_list[:-1]
        assert(len(stm_transcript_list) == len(res_list))

        self.online_alignment = alignment[0]
        self.stm_alignment = alignment[1]

        assert(self.online_alignment.replace("-", "") == online_transcript)
        assert(self.stm_alignment.replace("-", "") == stm_transcript_all)

        self.online_clean_to_alignment, self.online_alignment_to_clean = self._get_index_mapping(self.online_alignment)
        self.stm_clean_to_alignment, self.stm_alignment_to_clean = self._get_index_mapping(self.stm_alignment)

        self.online_transcript = online_transcript
        self.stm_transcript_all = stm_transcript_all
        self.stm_transcript_list = stm_transcript_list

        self.updated_res_list = self._get_punctuated_stm(res_list)

    def get_updated_res_list(self):
        return self.updated_res_list

    def _stm_index_to_online_index(self, stm_index, find_next_possible=False, find_previous_possible=False):
        align_index = self.stm_clean_to_alignment[stm_index]
        online_index = self.online_alignment_to_clean.get(align_index, None)

        if (online_index is None) and (find_next_possible or find_previous_possible):
            multiplier = 1 if find_next_possible else -1
            for i in range(1, 30):
                online_index = self.online_alignment_to_clean.get(align_index + multiplier * i, None)
                if online_index:
                    break
            if online_index is None:
                print("Failed to find next/previous alignment after 100 next attempt")
                raise GavinNoPuncForSTM

        return online_index, align_index

    def _get_punctuated_stm(self, res_list):

        current_index = 0
        for stm_i, stm_transcript in enumerate(self.stm_transcript_list):
            special_case = False
            substring_index_low = self.stm_transcript_all.find(stm_transcript, current_index)
            if substring_index_low == -1:
                raise ValueError("Substring Not Found")
            substring_index_high = substring_index_low + len(stm_transcript) - 1 # index of the last char in stm_transcript
            current_index = substring_index_high + 1 # index of the first char in next stm_transcript

            # Get the alignment index low and high, this applies to both online_alignment (non-exact) and stm_alignment (exact)
            online_index_low, align_index_low = self._stm_index_to_online_index(substring_index_low)
            online_index_high, align_index_high = self._stm_index_to_online_index(substring_index_high)

            # Try retrieving the punctuated transcript using online_index_low and online_index_high
            # There are multiple attempts, if all of them failed, we will fall back to use the stm transcript
            try:
                if online_index_low is None:
                    # Handle cases when current stm_transcript's first word is same as previous stm_transcript's last word
                    # We make the word belong to us
                    if stm_transcript.split(" ")[0] == self.stm_transcript_list[stm_i-1].split(" ")[-1]:
                        common_word = stm_transcript.split(" ")[0]
                        substring_index_low = self.stm_transcript_all.rfind(common_word, 0, substring_index_low)
                        online_index_low, align_index_low = self._stm_index_to_online_index(substring_index_low)
                        print("Case 11: {}".format(stm_transcript))
                        special_case = True
                    else: # Handle cases when stm_transcript contain extra word that is not in the online transcript, we choose to follow the online transcript
                        online_index_low, align_index_low = self._stm_index_to_online_index(substring_index_low, find_next_possible=True)
                        print("Case 12: {}".format(stm_transcript))
                        special_case = True
                if online_index_high is None:
                    # Handle cases when current stm_transcript's last word is same as next stm_transcript's first word
                    if len(self.stm_transcript_list) > stm_i+1 and stm_transcript.split(" ")[-1] == self.stm_transcript_list[stm_i+1].split(" ")[0]:
                        # We do not make the word belong to us, it should belong to the next stm_transcript
                        online_index_high, align_index_high = self._stm_index_to_online_index(substring_index_high, find_previous_possible=True)
                        print("Case 13: {}".format(stm_transcript))
                        special_case = True
                    else: # Handle cases when stm_transcript contain extra word that is not in the online transcript, we choose to follow the online transcript
                        online_index_high, align_index_high = self._stm_index_to_online_index(substring_index_high, find_previous_possible=True)
                        print("Case 14: {}".format(stm_transcript))
                        special_case = True

                # Increase online_index_high to include any following punctuation
                m = re.search("[ ,.?!]+", self.stm_transcript_all[online_index_high+1:])
                punc = m.group(0).rstrip(" ") if m else ""
                if punc:
                    print("Case 6: {}".format(punc))
                punc_stm_transcript = self.online_transcript[online_index_low:online_index_high+1] + punc
                if special_case:
                    print("Case 5: {}".format(punc_stm_transcript))
            except (GavinNoPuncForSTM) as e:
                # If error occurred, just use stm_transcript
                punc_stm_transcript = generate_clean_punc(stm_transcript, remove_trailing_space=True)
                print("Case 17: {} {}".format(punc_stm_transcript, type(e)))
                print("Case 5: {}".format(punc_stm_transcript))

            res_list[stm_i]["transcript"] = punc_stm_transcript

        return res_list

    def _get_index_mapping(self, alignment_string):
        dash_index = []
        original_index = 0
        clean_to_alignment = {}
        alignment_to_clean = {}
        for i, x in enumerate(alignment_string):
            if x == "-":
                dash_index.append(i)
            else:
                clean_to_alignment[original_index] = i
                alignment_to_clean[i] = original_index
                original_index += 1

        return clean_to_alignment, alignment_to_clean


def check_if_common_word_in_transcript(common_word, online_transcript):
    possible_insertions = [" ", " . ", " , ", " ! ", " ? "]
    found_in_online = False
    for ins in possible_insertions:
        if " " + common_word + ins + common_word + " " in online_transcript:
            found_in_online = True
    return found_in_online


def remove_duplicate_from_stm(stm_transcript_list, online_transcript):
    new_stm_transcript_list = []
    for stm_i, stm_transcript in enumerate(stm_transcript_list):
        stm_transcript_token = stm_transcript.split(" ")
        common_word_list = [i for i, j in zip(stm_transcript_token, stm_transcript_token[1:]) if i == j]
        if common_word_list:
            for common_word in common_word_list:
                found_in_online = check_if_common_word_in_transcript(common_word, online_transcript)
                if not found_in_online:
                    stm_transcript = stm_transcript.replace(common_word + " " + common_word, common_word)
        if len(stm_transcript_list) > stm_i + 1 and stm_transcript.split(" ")[-1] == stm_transcript_list[stm_i + 1].split(" ")[0]:
            # Do not remove if such duplicate words also in online transcript
            common_word = stm_transcript.split(" ")[-1]
            found_in_online = check_if_common_word_in_transcript(common_word, online_transcript)
            if not found_in_online:
                stm_transcript = stm_transcript.rsplit(" ", 1)[0]
        new_stm_transcript_list.append(stm_transcript)
    return new_stm_transcript_list


def prune_res_list(res_list, ran, one_word_or_letter):
    """
    Check if res_list have consecutive lines that have only 1 word. If so , remove it/
    :param res_list:
    :return:
    """
    new_res_list = []
    only_one = []
    for res_i, res in enumerate(res_list):
        if one_word_or_letter == "word":
            if len(res["transcript"].replace("<unk>", " ").replace("-", " ").lstrip(" ").rstrip(" ").split(" ")) == 1:
                only_one.append(True)
            else:
                only_one.append(False)
        else:
            if len(res["transcript"].replace("<unk>", " ").replace("-", " ").lstrip(" ").rstrip(" ").split(" ")) == 1 and \
                    len(res["transcript"].replace("<unk>", " ").replace("-", " ").lstrip(" ").rstrip(" ").split(" ")[0]) == 1:
                only_one.append(True)
            else:
                only_one.append(False)

    to_remove = set()
    for res_i, bol in enumerate(only_one):
        consecutive_only_one_word = True
        try:
            for i in range(ran):
                consecutive_only_one_word = consecutive_only_one_word and only_one[res_i+i]
            if consecutive_only_one_word:
                to_remove.update(range(res_i, res_i+ran))
        except IndexError:
            break

    if to_remove:
        for res_i, res in enumerate(res_list):
            if res_i not in to_remove:
                new_res_list.append(res)
        return new_res_list
    else:
        return res_list


def update_res_list_mass_process(res_list, punc_txt_path):
    with open(punc_txt_path) as fh:
        online_transcript = fh.read().rstrip("\n").lstrip(" ").rstrip(" ")

    res_list = prune_res_list(res_list, 4, one_word_or_letter="word")
    res_list = prune_res_list(res_list, 2, one_word_or_letter="letter")

    # Collect individual transcript in a list
    stm_transcript_list = []
    for res in res_list:
        stm_transcript = res["transcript"]
        stm_transcript = stm_transcript.replace("<unk>", " ").replace("-", " ").lstrip(" ").rstrip(" ")
        stm_transcript = re.sub("\s+", " ", stm_transcript)
        stm_transcript_list.append(stm_transcript)

    # Remove duplicate common word between two consecutive stm transcripts
    # Or remove duplicate consecutive word in one stm transcript
    stm_transcript_list = remove_duplicate_from_stm(stm_transcript_list, online_transcript)

    stm_transcript_all = " ".join(stm_transcript_list)

    # Get entire alignment and print it
    alignment = find_best_alignment(online_transcript, stm_transcript_all)
    print(format_alignment(*alignment))

    aa = AugmentedAlignment(alignment, online_transcript, stm_transcript_all, stm_transcript_list, res_list)

    updated_res_list = aa.get_updated_res_list()
    return updated_res_list


def get_mergable(mergable_list, index, upper_threshold):
    prev_merge = float("Inf")
    next_merge = float("Inf")

    if index > 0 and mergable_list[index-1][1] >= mergable_list[index][0]:
        prev_merge = mergable_list[index][1] - mergable_list[index-1][0]
    if len(mergable_list) > index+1 and mergable_list[index][1] >= mergable_list[index+1][0]:
        next_merge = mergable_list[index+1][1] - mergable_list[index][0]

    if prev_merge == next_merge == float("Inf"):
        not_mergable_item = mergable_list.pop(index)
        return mergable_list, not_mergable_item

    if prev_merge < upper_threshold or next_merge < upper_threshold:
        if prev_merge <= next_merge:
            mergable_list[index-1] = [mergable_list[index-1][0], mergable_list[index][1], mergable_list[index-1][2] + mergable_list[index][2]]
        else:
            mergable_list[index+1] = [mergable_list[index][0], mergable_list[index+1][1], mergable_list[index][2] + mergable_list[index+1][2]]
        del mergable_list[index]
        return mergable_list, None
    else:
        not_mergable_item = mergable_list.pop(index)
        return mergable_list, not_mergable_item

def assert_mergable_sequence(mergable_list):
    start_sequence = [m[0] for m in mergable_list]
    end_sequence = [m[1] for m in mergable_list]
    index_sequence = [m[2][0] for m in mergable_list]

    assert(start_sequence==sorted(start_sequence))
    assert(end_sequence==sorted(end_sequence))
    assert(index_sequence==sorted(index_sequence))

def merge_time_tuple_list(mergable_list, upper_threshold):
    assert_mergable_sequence(mergable_list)

    last_min = None
    current_min = False
    not_mergable_list = []

    while True:
        if last_min == current_min or not mergable_list:
            break
        last_min = current_min
        current_min = min(mergable_list, key=lambda x: x[1] - x[0])
        current_min_index = mergable_list.index(current_min)

        mergable_list, not_mergable_item = get_mergable(mergable_list, current_min_index, upper_threshold)
        assert_mergable_sequence(mergable_list)
        if not_mergable_item:
            not_mergable_list.append(not_mergable_item)

    assert(not mergable_list)
    return sorted(not_mergable_list)

def merge_stm(res_list, upper_threshold=30):
    """
    Merge transcripts in stm together. Up to an upper threshold of "upper_threshold" seconds
    """

    mergable_list = [[res["start_time"], res["end_time"], [res_i]] for res_i, res in enumerate(res_list)]
    merged_time_list = merge_time_tuple_list(mergable_list, upper_threshold)

    new_res_list = []

    for time_slot in merged_time_list:
        start_time = time_slot[0]
        end_time = time_slot[1]
        indices = time_slot[2]
        index_0 = indices[0]

        transcript = ""
        for index in indices:
            transcript += res_list[index]["transcript"] + " "
        transcript = transcript.rstrip(" ")

        res = {"start_time": start_time, "end_time": end_time, "filename": res_list[index_0]["filename"],
               "transcript": transcript, "token_1": res_list[index_0]["token_1"],
               "token_2": res_list[index_0]["token_2"], "token_5": res_list[index_0]["token_5"]}

        new_res_list.append(res)

    return new_res_list


def create_puncstm(ted_dir, converted_dir, html_dir, punc_dir, rawpunc_dir, plain_dir, stm_dir, puncstm_dir, not_merged_stm_dir):
    # We create stm with punc for all clean punc available
    punc_txt_list = os.listdir(punc_dir)
    punc_txt_list.reverse()
    punc_txt_list_full_path = [os.path.join(punc_dir, punc_txt) for punc_txt in punc_txt_list]

    for punc_txt, punc_txt_full_path in zip(punc_txt_list, punc_txt_list_full_path):
        try:
            speaker = punc_txt.split(".")[0]

            # if not speaker in ("VijayKumar_2012", "BlaiseAguerayArcas_2010", "DavidBlaine_2009P", "ToddScott_2017S", "JoachimdePosada_2009U", "AllanJones_2011G", "NatalieMacMaster_2003", "NateSilver_2009", "BernieKrause_2013G", "EstherPerel_2013S", "EvelynGlennie_2003", "Rives4AM_2007"):
            #     continue

            print("Current Speaker is: {}".format(speaker))
            if speaker in ["WadeDavis_2003", "BillGates_2010"]:
                continue
            original_stm = os.path.join(stm_dir, speaker + ".stm")
            new_stm = os.path.join(puncstm_dir, speaker + ".stm")
            not_merged_stm = os.path.join(not_merged_stm_dir, speaker + ".stm")

            #if not (os.path.exists(not_merged_stm) and os.path.exists(new_stm)):
            if True:
                # update transcript section in res_list
                # try:
                res_list = get_utterances_from_stm(original_stm)

                # # Special edge cases:
                # if speaker == "NicoleParis_2015Y":
                #     res_list = res_list[:-1]
                # if speaker == "ClaronMcFadden_2010X":
                #     res_list = res_list[:-5]
                # if speaker == "AdamOckelford_2013X":
                #     res_list = res_list[:-2]
                # if speaker == "ItayTalgam_2009G":
                #     res_list = res_list[:-2]
                # if speaker == "JiHaePark_2013":
                #     res_list = res_list[:-2]
                # if speaker == "YvesBehar_2008":
                #     res_list = res_list[:-1]
                # if speaker == "EddiReader_KiteflyersHill_2004":
                #     res_list = res_list[:-6]
                # if speaker == "JohnBohannon_2011X":
                #     res_list = res_list[:-1]
                # if speaker == "JenniferLin_2004":
                #     res_list = res_list[:-8]
                # if speaker == "BrianGoldman_2011X":
                #     res_list[-1]["transcript"] = res_list[-1]["transcript"].replace(" applause", "")

                if speaker == "ClaronMcFadden_2010X":
                    res_list = res_list[:-4]
                if speaker == "AdamOckelford_2013X":
                    res_list = res_list[:-1]
                if speaker == "ItayTalgam_2009G":
                    res_list = res_list[:-1]
                if speaker == "JiHaePark_2013":
                    res_list = res_list[:-1]
                if speaker == "EddiReader_KiteflyersHill_2004":
                    res_list = res_list[:-5]
                if speaker == "JenniferLin_2004":
                    res_list = res_list[:-7]


                res_list = update_res_list_mass_process(res_list, punc_txt_full_path)

                write_stm_from_utterances(not_merged_stm, res_list)
                # except:
                #     print("Speaker: {} suffered error, please debug".format(speaker))

                res_list = merge_stm(res_list, 30)
                write_stm_from_utterances(new_stm, res_list)
        except:
            pass

def create_html_and_punc(html_dir, punc_dir, plain_dir, stm_dir, rawpunc_dir, puncstm_dir, speaker):
    """
    For "speaker", fetch the best corresponding transcript from ted.com, and clean the text in the transcript.
    Stores:plain transcript file based on stm content in plain_dir, this is used to determine the best matching html
           html information in html_dir, this is to avoid repetitively fetching information from online
           raw transcript with punctuation downloaded from the best ted.com webpage, stored in rawpunc_dir
           clean raw transcript and store the result in punc_dir

    :param html_dir:
    :param punc_dir:
    :param plain_dir:
    :param stm_dir:
    :param rawpunc_dir:
    :param puncstm_dir:
    :param speaker:
    :return: None
    """

    # Perform link reading and html conversion if raw_transcript does not exist
    try:
        if not os.path.exists(os.path.join(punc_dir, speaker + ".txt")):
            # exceptions, some speakers result in error
            if speaker == 'KarenThompsonWalker_2012G':
                return
            # Convert stm into a plain file of transcription (the transcription does not have punctuation or capitalization)
            plain_file = get_plain_with_stm(plain_dir, stm_dir, speaker)
            # Get the html_file that contains transcript that is the closest to the plain file (if there is anything close)
            best_link = write_html_with_name(html_dir, plain_file, speaker)

            # Write a punctuated transcript into punc_dir
            raw_transcript = write_raw_punc_with_name(rawpunc_dir, best_link, speaker)

            write_clean_punc_with_name(punc_dir, rawpunc_dir, speaker)
    except:
        pass

def download_and_preprocess_online_transcript(ted_dir, html_dir, punc_dir, rawpunc_dir, plain_dir, stm_dir, puncstm_dir):
    """
    Collect the sph and stm file path for all speakers. The sph and stm files are from the TEDLIUM dataset.
    :param ted_dir: directory after unpacking the TEDLIUM_release2.tar.gz official resource
    :param html_dir: directory containing processed information of fetched html, this is to avoid repetitive access of internet
    :param punc_dir:
    :param rawpunc_dir:
    :param plain_dir:
    :param stm_dir: directory containing the stm files from official ted resourse
    :param puncstm_dir:
    :return:
    """

    # Collect speaker names, sph, stm file paths
    entries = os.listdir(os.path.join(ted_dir, "sph"))
    all_files = {}
    for sph_file in entries:
        speaker_name = sph_file.split('.sph')[0]

        sph_file_full = os.path.join(ted_dir, "sph", sph_file)
        stm_file_full = os.path.join(ted_dir, "stm", "{}.stm".format(speaker_name))

        assert os.path.exists(sph_file_full) and os.path.exists(stm_file_full)

        assert speaker_name not in all_files
        all_files[speaker_name] = {"sph": sph_file_full, "stm": stm_file_full}

    for speaker_name, other_info in all_files.items():
        if DEBUG:
            try:
                create_html_and_punc(html_dir, punc_dir, plain_dir, stm_dir, rawpunc_dir, puncstm_dir, speaker_name)
            except CustomWebErrorNoOnlineTranscript:
                print("No Transcript has been found online, processing failed for: {}".format(speaker_name))
                continue
        else:
            try:
                create_html_and_punc(html_dir, punc_dir, plain_dir, stm_dir, rawpunc_dir, puncstm_dir, speaker_name)
            except Exception as e:
                if isinstance(e, CustomWebErrorNoOnlineTranscript):
                    write_unique_msg_to_file(error_no_online_transcript_log, speaker_name + " " + e.args[0])
                elif isinstance(e, CustomWebError404):
                    pass
                else:
                    print("Unexpected error occurred: {}, for {}".format(e, speaker_name))
                    write_unique_msg_to_file(error_all_others_log, speaker_name)


def prepare_punc_dir(ted_dir):
    """
    Go through each speaker in the directory. Call each speaker with create_html_and_punc.
    Continue to next to speaker if any error occurred. (And log the failed speaker).
    """
    converted_dir = os.path.join(ted_dir, "converted")

    html_dir = create_dir(converted_dir, "html")
    punc_dir = create_dir(converted_dir, "punc")
    rawpunc_dir = create_dir(converted_dir, "rawpunc")
    plain_dir = create_dir(converted_dir, "plain")
    stm_dir = os.path.join(ted_dir, "stm")
    not_merged_stm_dir = create_dir(converted_dir, "notmergedstm")
    puncstm_dir = create_dir(converted_dir, "puncstm")

    download_and_preprocess_online_transcript(ted_dir, html_dir, punc_dir, rawpunc_dir, plain_dir, stm_dir, puncstm_dir)
    create_puncstm(ted_dir, converted_dir, html_dir, punc_dir, rawpunc_dir, plain_dir, stm_dir, puncstm_dir, not_merged_stm_dir)

def download_and_extract():
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

    target_unpacked_dir = os.path.join(target_dl_dir, "TEDLIUM_release2")
    if args.tar_path and os.path.exists(args.tar_path):
        target_file = args.tar_path
    else:
        print("Could not find downloaded TEDLIUM archive, Downloading corpus...")
        wget.download(TED_LIUM_V2_DL_URL, target_dl_dir)
        target_file = os.path.join(target_dl_dir, "TEDLIUM_release2.tar.gz")

    if not os.path.exists(target_unpacked_dir):
        print("Unpacking corpus...")
        tar = tarfile.open(target_file)
        tar.extractall(target_dl_dir)
        tar.close()
    else:
        print("Found TEDLIUM directory, skipping unpacking of tar files")

def main():
    """
    Create TEDLIUM data with continuous audio (train, validation and test).
    Webscraping from ted.com and google is required.
    """
    target_dl_dir = "TEDLIUM_dataset/"
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)

    target_unpacked_dir = os.path.join(target_dl_dir, "TEDLIUM_release2")

    train_ted_dir = os.path.join(target_unpacked_dir, "train")
    val_ted_dir = os.path.join(target_unpacked_dir, "dev")
    test_ted_dir = os.path.join(target_unpacked_dir, "test")

    prepare_punc_dir(train_ted_dir)
    prepare_punc_dir(val_ted_dir)
    prepare_punc_dir(test_ted_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processes and downloads TED-LIUMv2 dataset.')
    parser.add_argument("--target-dir", default='TEDLIUM_dataset/', type=str, help="Directory to store the dataset.")
    parser.add_argument("--tar-path", default='TEDLIUM_dataset/TEDLIUM_release2.tar.gz',
                        type=str, help="Path to the TEDLIUM_release tar if downloaded (Optional).")
    args = parser.parse_args()

    TED_LIUM_V2_DL_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"
    main()
