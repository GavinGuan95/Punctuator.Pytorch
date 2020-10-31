import unicodedata
import io
from Bio import pairwise2
from Bio.pairwise2 import format_alignment


def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            tokens = stm_line.split()
            start_time = float(tokens[3])
            end_time = float(tokens[4])
            filename = tokens[0]
            transcript = unicodedata.normalize("NFKD",
                                               " ".join(t for t in tokens[6:]).strip()). \
                encode("utf-8", "ignore").decode("utf-8", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                res.append({
                    "start_time": start_time, "end_time": end_time,
                    "filename": filename, "transcript": transcript
                })
        return res

clean_transcript = " in the next six minutes that you will listen to me , the world will have lost three mothers while delivering their babies one , because of a severe complication second , because she will be a teenager and her body will not be prepared for birth but the third , only because of lack of access to basic clean tools at the time of childbirth she will not be alone over one million mothers and babies die every single year in the developing world , only because of lack of access to basic cleanliness while giving birth to their babies my journey began on a hot summer afternoon in india in two thousand and eight , when after a day of meeting women and listening to their needs , i landed in a thatched hut with a midwife as a mother , i was very curious on how she delivered babies in her house after a deep and engaging conversation with her on how she considered it a profound calling to do what she was doing , i asked her a parting question do you have the tools that you need to deliver the babies ? i got to see her tool , this is what i use to separate the mother and the baby , she said unsure of how to react , i held this agricultural tool in my hand in shock i took a picture of this , hugged her and walked away my mind was flooded with reflections of my own infection that i had to struggle with for a year past childbirth despite having access to the best medical care , and memories of my conversation with my father , who had lost his mom to childbirth , on how he thought his life would be so different if she would have been just next to him growing up as a product developer , i started my process of research i was very excited to find that there was a product out there called the clean birth kit but i just couldn 't buy one for months they were only assembled based on availability of funding finally , when i got my hands on one , i was in shock again i would never use these tools to deliver my baby , i thought but to confirm my instincts , i went back to the women , some of whom had the experience of using this product lo and behold , they had the same reaction and more the women said they would rather deliver on a floor than on a plastic sheet that smeared blood all over they were absolutely right , it would cause more infection the thread provided was a highway to bacterial infection through the baby 's umbilical cord , and the blade used was the kind that men used for shaving , and they did not want it anywhere close to them there was no incentive for anybody to redesign this product , because it was based on charity the women were never consulted in this process and to my surprise , the need was not only in homes but also in institutional settings with high volume births situations in remote areas were even more daunting this had to change i made this my area of focus i started the design process by collecting feedback , developing prototypes and engaging with various stakeholders researching global protocols with every single prototype , we went back to the women to ensure that we had a product for them what i learned through this process was that these women , despite their extreme poverty , placed great value on their health and well being they were absolutely not poor in mind as with all of us , they would appreciate a well designed product developed for their needs after many iterations working with experts , medical health professionals and the women themselves , i should say it was not an easy process at all , but we had a simple and beautiful design for a dollar more than what the existing product was offered for , at three dollars , we were able to deliver , janma , a clean birth kit in a purse janma , meaning , birth , contained a blood absorbing sheet for the woman to give birth on , a surgical scalpel , a cord clamp , a bar of soap , a pair of gloves and the first cloth to wipe the baby clean all this came packaged in a beautiful purse that was given to the mother as a gift after all her hard work , that she carried home with pride as a symbol of prosperity one woman reacted to this gift she said , is this really mine ? can i keep it ? the other one said , will you give me a different color when i have my next baby ? better yet , a woman expressed that this was the first purse that she had ever owned in her life the kit , aside from its symbolism and its simplicity , is designed to follow globally recommended medical protocol and serves as a behavior change tool to follow steps one after the other it can not only be used in homes , but also in institutional settings to date , our kit has impacted over six hundred thousand mothers and babies around the world it 's a humbling experience to watch these numbers grow , and i cannot wait until we reach a hundred million but women 's health issues do not end here there are thousands of simple issues that require low cost interventions we have facts to prove that if we invest in women and girls and provide them with better health and well being , they will deliver healthier and wealthier and prosperous communities we have to start by bringing simplicity and dignity to women 's health issues from reducing maternal mortality , to breaking taboos , to empowering women to take control of their own lives this is my dream but it is not possible to achieve it without engaging men and women alike from around the world , yes , all of you i recently heard this lyric by leonard cohen , ring the bells that still can ring forget your perfect offering there is a crack in everything that 's how the light gets in , this is my bit of light but we need more light in fact , we need huge spotlights placed in the world of women 's health if we need a better tomorrow we should never forget that women are at the center of a sustainable world , and we do not exist without them thank you  "
stm_z = "/home/guanyush/Downloads/deepspeech/data/TEDLIUM_dataset/TEDLIUM_release3_full/data/stm/ZubaidaBai_2016S.stm"


utt = get_utterances_from_stm(stm_z)
print(utt)

stm_transcripts = [ele['transcript'] for ele in utt]
stm_transcripts = [ele.replace("<unk>", "") for ele in stm_transcripts]

print(stm_transcripts)


for stm_transcript in stm_transcripts:

    alignments = pairwise2.align.globalms(clean_transcript, stm_transcript, 2, -1, -1, -0.01)

    for a in alignments:
        print(format_alignment(*a))

