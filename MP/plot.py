import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np

# Configure font sizes and family for consistent plotting
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# These lists will store measured values for plotting
distributed_MP_values = []
contiguous_MP_values = []
contiguous_SM_increasing_MP = []
distributed_SM_increasing_MP = []

# Load data for contiguous SMs with increasing MP count
for mp_count in range(1, 5):
    df = pd.read_csv(f'GPC0_SM14_MP{mp_count}.log',
                     delim_whitespace=True,
                     header=None,
                     skiprows=10)
    if df.shape[0] > 0:
        # Extract numeric part from the last column of the first row
        value_str = str(df.iloc[0, -1])
        numeric_str = re.findall(r'(\d+\.?\d*)', value_str)[0]
        contiguous_SM_increasing_MP.append(pd.to_numeric(numeric_str))

# Load data for distributed SMs with increasing MP count
for mp_count in range(1, 5):
    df = pd.read_csv(f'distributedSM14_MP{mp_count}.log',
                     delim_whitespace=True,
                     header=None,
                     skiprows=10)
    if df.shape[0] > 0:
        value_str = str(df.iloc[0, -1])
        numeric_str = re.findall(r'(\d+\.?\d*)', value_str)[0]
        distributed_SM_increasing_MP.append(pd.to_numeric(numeric_str))

# Load data for distributed 14/28 SMs, 1 MP using file paths
file_paths_distributed_sm = ['distributedSM14_MP1.log','distributedSM28_MP1.log']

for file_path in file_paths_distributed_sm:
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
        skiprows=10,
        on_bad_lines='skip'
    )
    if df.shape[0] > 0:
        value_str = str(df.iloc[0, -1])
        numeric_str = re.findall(r'(\d+\.?\d*)', value_str)[0]
        distributed_MP_values.append(pd.to_numeric(numeric_str))


# Load data for contiguous 14/28 SMs, 1 MP
file_paths = ['GPC0_SM14_MP1.log', 'contiguousSM28_MP1.log']
for file_path in file_paths:
    df = pd.read_csv(file_path,
                     delim_whitespace=True,
                     skiprows=10,
                     header=None,
                     on_bad_lines='skip')
    if df.shape[0] > 0:
        value_str = str(df.iloc[0, -1])
        numeric_str = re.findall(r'(\d+\.?\d*)', value_str)[0]
        contiguous_MP_values.append(pd.to_numeric(numeric_str))

# Start creating plots
plt.figure(figsize=(4.72, 1.77))  # in inches (12cm x 4.5cm)

# Subplot 1: Compare contiguous vs distributed MP
plt.subplot(1, 2, 1)
x_positions = np.arange(1, 3)  # for two bars
bar_width = 0.35

# Contiguous MP bars
plt.bar(x_positions - bar_width/2,
        contiguous_MP_values,
        bar_width,
        color='#404040',
        edgecolor='black',
        linewidth=1,
        label='Contiguous MP')

# Distributed MP bars
plt.bar(x_positions + bar_width/2,
        distributed_MP_values,
        bar_width,
        color='#808080',
        edgecolor='black',
        linewidth=1,
        label='Distributed MP')

plt.ylim(0, 800)
plt.yticks(range(0, 801, 200))
plt.xticks([1, 2], ['14', '28'])
plt.xlabel('# of SMs')

# Subplot 2: Compare contiguous vs distributed SM for multiple MP counts
plt.subplot(1, 2, 2)
x_positions = np.arange(1, 5)

# Contiguous SM bars
plt.bar(x_positions - bar_width/2,
        contiguous_SM_increasing_MP,
        bar_width,
        color='#404040',
        edgecolor='black',
        linewidth=1,
        label='Contiguous SM')

# Distributed SM bars
plt.bar(x_positions + bar_width/2,
        distributed_SM_increasing_MP,
        bar_width,
        color='#808080',
        edgecolor='black',
        linewidth=1,
        label='Distributed SM')

plt.ylim(0, 800)
plt.yticks(range(0, 801, 200))
plt.xticks([1, 2, 3, 4], ['1', '2', '3', '4'])
plt.xlabel('# of MPs')

# Adjust spacing and add legend
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=0.8, wspace=0.4)
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles,
                    labels,
                    loc='lower center',
                    bbox_to_anchor=(-0.1, 1),
                    fancybox=False,
                    shadow=False,
                    ncol=1,
                    fontsize=12)
legend.get_frame().set_edgecolor('black')

# Save the figure
plt.savefig('plot.png', bbox_inches='tight', transparent=False, dpi=300)