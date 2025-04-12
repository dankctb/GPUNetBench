import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# Define the file names
files = ['V100', 'A100', 'H100']

subscript_map = {
    '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
    '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
    'l': 'ₗ', 'g': '₉'  # 'ₗ' is subscript for "l"
}

def to_subscript(text):
    return ''.join(subscript_map.get(char, char) for char in text)

# Define the x-axis labels for each file
x_labels_dict = {
    'V100': ['TPC', r'GPC$_l$', r'GPC$_g$'],
    'A100': ['TPC', r'GPC$_l$', r'GPC$_g$'],
    'H100': ['TPC', 'CPC', r'GPC$_l$', r'GPC$_g$']
}

# Define the width of the bars
width = 0.3

# Create the figure with specified size (converted from cm to inches)
fig = plt.figure(figsize=(16/2.54, 4/2.54))

# Define the grid
gs = gridspec.GridSpec(1, 3, width_ratios=[1.1, 1, 1.4]) 

for i, file in enumerate(files):
    # Initialize lists to store the data
    first_col = []
    last_col = []
    # Adjust the space between the subplots
    plt.subplots_adjust(wspace=0)

    # Open and read the file
    with open(file, 'r') as f:
        for line in f:
            # Split the line into columns
            columns = line.split(',')
            
            # Add the first and last column to the respective lists
            first_col.append(float(columns[0]))
            last_col.append(float(columns[-1]))

    # Define the x-axis labels
    x_labels = [''] + x_labels_dict[file] + ['']

    # Define the x locations for the groups
    x = np.arange(len(x_labels))

    # Create the subplot
    ax = plt.subplot(gs[i])

    # Create the bar plot
    ax.bar(x[1:-1] - width/2, first_col, width, label='Read', color='lightgrey', edgecolor='black')
    ax.bar(x[1:-1] + width/2, last_col, width, label='Write', color='grey', edgecolor='black')

    # Add y-axis title to the first subplot
    if i == 0:
        ax.set_ylabel('Speedup')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=50)

    # Set y-axis limit and ticks
    ax.set_ylim([0, 14])
    if (i==0):
        ax.set_yticks(np.arange(0, 15, 2))
        ax.set_yticks(np.arange(0, 14, 1), minor=True)
    if (i > 0):
        ax.set_yticks([])
        ax.set_yticklabels([])


    # Write the file name under the plot
    ax.set_xlabel(file)

# Add a legend
legend = plt.legend(loc='lower center', bbox_to_anchor=(-1.1, 0.35), fancybox=False, shadow=False, ncol=1, columnspacing = 0.5,borderpad= 0.5 , handlelength=1)
frame = legend.get_frame()
frame.set_edgecolor('black')

# Save the figure
plt.savefig("cpc_bw.pdf", format='pdf', bbox_inches='tight', transparent=True)

# Show the plot
plt.show()

# Close the figure
plt.close()