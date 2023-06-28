# import numpy as np
# import matplotlib.pyplot as plt
#
# success_rates =  np.array([1.0000, 0.9062, 0.7647, 0.9062, 0.7368, 0.5625, 0.9667, 0.6444, 0.6341,
#         0.9667, 0.6875, 0.8710, 0.7714, 0.4878, 0.7027, 0.6333, 0.7273, 0.8667,
#         0.3659, 0.9032, 0.6316, 0.6111, 0.9333, 0.7381, 0.6923, 0.5854, 0.3559,
#         0.6829, 0.0784, 0.7097, 0.5429, 0.4250, 0.4000, 0.5714, 0.6047, 0.0000,
#         0.5714, 0.2093, 0.7273, 0.5882, 0.6667, 0.1053, 0.1000, 0.2667, 0.0000,
#         0.0000, 0.0000, 0.1967, 0.0000]).reshape(7, 7)
#
#
#
# # Generate a random 2D array
# arr = np.flip(success_rates, axis=0)
#
#
# # Calculate the row and column averages
# row_means = np.mean(arr, axis=1)
# col_means = np.mean(arr, axis=0)
#
# # Create a figure and axis
# # fig, ((ax, row), (column, all_mean)) = plt.subplots(2, 2)
#
# unit = 10
# gap = 2
# side = unit * 8 + gap
#
#
# fig = plt.figure()
# ax = plt.subplot2grid((side, side), (0, 0), rowspan=unit * 7, colspan=unit * 7)
# row = plt.subplot2grid((side, side), (0, 7 * unit + gap), rowspan=7 * unit, colspan=unit)
# column = plt.subplot2grid((side, side), (7 * unit + gap, 0), rowspan=unit, colspan=7 * unit)
# all_mean = plt.subplot2grid((side, side), (7 * unit + gap, 7 * unit + gap), rowspan=unit, colspan=unit)
#
# im_row = row.imshow(row_means.reshape(7, 1), cmap='viridis')
# im_column = column.imshow(col_means.reshape(1, 7), cmap='viridis')
# im_all_mean = all_mean.imshow(np.array([np.mean(arr)]).reshape(1,1), cmap='viridis')
#
# row.axis('off')
# column.axis('off')
# all_mean.axis('off')
#
# # Create a heatmap using imshow
# im = ax.imshow(arr, cmap='viridis')
#
# # Add a colorbar
# # cbar = ax.figure.colorbar(im, ax=ax)
#
# x_labels = ['0', '1', '2', '3', '4', '5', '6']
# y_labels = reversed(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
#
# # Set the ticks and tick labels for the row and column averages
# ax.set_xticks(np.arange(len(col_means)))
# ax.set_yticks(np.arange(len(row_means)))
# ax.set_xticklabels(x_labels)
# ax.set_yticklabels(y_labels)
#
# ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
#
# # Rotate the tick labels and set their alignment
# plt.setp(ax.get_xticklabels(),
#          rotation_mode="anchor")
#
# # Loop over the data and add annotations to the heatmap
# for i in range(len(row_means)):
#     for j in range(len(col_means)):
#         text = ax.text(j, i, arr[i, j], ha="center", va="center", color="w")
#
# # Add a title
# # ax.set_title("Success Rates of Relocation on the EGAD Test Set")
#
# # Show the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

unit = 10
gap = 4
side = unit * 8 + gap * 5

long = (7 * unit + gap) / side
short = unit / side
space = gap / side

success_rates_1h = np.array(
[0.9301, 0.8800, 0.7923, 0.9031, 0.7752, 0.6402, 0.9351, 0.6371, 0.6681,
        0.9412, 0.6601, 0.9067, 0.7512, 0.5261, 0.7321, 0.6630, 0.6376, 0.9061,
        0.3600, 0.8703, 0.5468, 0.6256, 0.9185, 0.7729, 0.6538, 0.6230, 0.3468,
        0.6214, 0.1355, 0.7081, 0.5192, 0.4250, 0.4895, 0.5909, 0.5953, 0.0071,
        0.5813, 0.1861, 0.6181, 0.6618, 0.5282, 0.1585, 0.1556, 0.3260, 0.0159,
        0.0039, 0.0052, 0.2170, 0.0154]).reshape(7, 7)

success_rates_10min =  np.array([1.0000, 0.9062, 0.7647, 0.9062, 0.7368, 0.5625, 0.9667, 0.6444, 0.6341,
        0.9667, 0.6875, 0.8710, 0.7714, 0.4878, 0.7027, 0.6333, 0.7273, 0.8667,
        0.3659, 0.9032, 0.6316, 0.6111, 0.9333, 0.7381, 0.6923, 0.5854, 0.3559,
        0.6829, 0.0784, 0.7097, 0.5429, 0.4250, 0.4000, 0.5714, 0.6047, 0.0000,
        0.5714, 0.2093, 0.7273, 0.5882, 0.6667, 0.1053, 0.1000, 0.2667, 0.0000,
        0.0000, 0.0000, 0.1967, 0.0000]).reshape(7, 7)

arr = np.flip(success_rates_1h, axis=0)

# Calculate the row and column averages
row_means = np.mean(arr, axis=1)
col_means = np.mean(arr, axis=0)

# # Create a figure with four subplots
# fig = plt.figure(figsize=(1, 1))
# # fig = plt.figure()
#
# # # Create the top-left subplot with a size of 7x7
# # ax1 = fig.add_subplot(221)
# # ax1.imshow(col_means.reshape(1, 7), cmap='viridis')
# # ax1.axis('off')
# #
# # # Create the top-right subplot with a size of 7x1
# # ax2 = fig.add_subplot(222)
# # ax2.imshow(np.array([np.mean(arr)]).reshape(1,1), cmap='viridis')
# # ax2.axis('off')
#
# # Create the bottom-left subplot with a size of 1x7
# ax3 = fig.add_subplot(121)
# ax3.imshow(arr, cmap='viridis')
# ax3.axis('off')
#
# # Create the bottom-right subplot with a size of 1x1
# ax4 = fig.add_subplot(122)
# ax4.imshow(row_means.reshape(7, 1), cmap='viridis')
# ax4.axis('off')
#
# # Adjust the spacing between subplots
# fig.subplots_adjust(hspace=0.2, wspace=0.2)
#
# r = 0.9
#
# # Set the sizes of the subplots
# # ax1.set_position([space, space, long, short])
# # ax2.set_position([long + space * 2, space, short, short])
# ax3.set_position([space, space, long * r, long * r])
# ax4.set_position([long + space * 2, space, short, long])
#
# # Show the plot
# plt.show()




#

# Create a figure and axis
# fig, ((ax, row), (column, all_mean)) = plt.subplots(2, 2)

unit = 10
gap = 2
side = unit * 8 + gap


# fig = plt.figure()
# ax = plt.subplot2grid((side, side), (0, 0), rowspan=unit * 7, colspan=unit * 7)
# row = plt.subplot2grid((side, side), (0, 7 * unit + gap), rowspan=7 * unit, colspan=unit)
# column = plt.subplot2grid((side, side), (7 * unit + gap, 0), rowspan=unit, colspan=7 * unit)
# all_mean = plt.subplot2grid((side, side), (7 * unit + gap, 7 * unit + gap), rowspan=unit, colspan=unit)

# im_row = row.imshow(row_means.reshape(7, 1), cmap='viridis')
# # im_column = column.imshow(col_means.reshape(1, 7), cmap='viridis')
# # im_all_mean = all_mean.imshow(np.array([np.mean(arr)]).reshape(1,1), cmap='viridis')
#
# row.axis('off')
# column.axis('off')
# all_mean.axis('off')

fig, ax = plt.subplots()


# Create a heatmap using imshow
im = ax.imshow(arr, cmap='viridis')

# Add a colorbar
cbar = ax.figure.colorbar(im, ax=ax)

x_labels = ['0', '1', '2', '3', '4', '5', '6']
y_labels = reversed(['A', 'B', 'C', 'D', 'E', 'F', 'G'])

# Set the ticks and tick labels for the row and column averages
ax.set_xticks(np.arange(len(col_means)))
ax.set_yticks(np.arange(len(row_means)))
ax.set_xticklabels(x_labels, fontsize='large')
ax.set_yticklabels(y_labels, fontsize='large')

# ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(),
         rotation_mode="anchor")

# Loop over the data and add annotations to the heatmap
for i in range(len(row_means)):
    for j in range(len(col_means)):
        if arr[i, j] > 0.5:
            color = 'black'
        else:
            color = 'white'
        text = ax.text(j, i, "{:.2f}".format(arr[i, j]), ha="center", va="center", color=color, fontsize='large')

# Add a title
# ax.set_title("Success Rates of Relocation on the EGAD Test Set")

ax.set_xlabel('Complexity', fontsize='x-large')
ax.set_ylabel('Difficulty', fontsize='x-large')

# Show the plot
# plt.show()
#
#
#
#
# # Generate some random data
# data = np.random.randn(100)
#
plt.savefig('success_rates.png', dpi=400)