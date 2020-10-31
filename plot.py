import matplotlib.pyplot as plt
loss_list = {}
count_list = {}
with open("gavin_train.log") as fh:
    for line in fh:
        try:
            loss = line.split()[9]
            epoch = line.split()[1].lstrip("[").split("]")[0]
            if epoch in loss_list:
                loss_list[epoch] += float(loss)
                count_list[epoch] += 1
            else:
                loss_list[epoch] = float(loss)
                count_list[epoch] = 1
        except ValueError:
            dummy = 1

print(loss_list)
print(count_list)


loss = []

for i in range(1, 71):
    ii = str(i)
    loss.append(loss_list[ii] / count_list[ii])

print(loss)


fig = plt.figure()

plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
