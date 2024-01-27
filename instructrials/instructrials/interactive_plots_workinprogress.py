#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output
import seaborn as sns

from pynteractive.importer import import_dataframe, read_csv_as_list

# Global variable to store the color mapping for each group.
group_color_mapping = {}


def get_distinct_colors(n):
    """
    Returns n equally spaced colors from the HSV colormap in RGBA format.
    
    Parameters:
    - n: Number of colors
    
    Returns:
    - List of RGBA tuples
    """
    cmap = plt.cm.hsv
    hues = np.linspace(0, 1, n + 1, endpoint=False)[:-1]  # Avoid endpoint and take n hues
    return  [cmap(h) for h in hues]

def color_generator(n):
    """Yield colors one by one from a list of n distinct colors."""
    colors = get_distinct_colors(n)
    for color in colors:
        yield color


def plot_avg_scores_by_group_with_variation(ax, df, task_order, groupby_col,
                                            show_sem, show_std, show_all,
                                            title, xlabel, ylabel, legend_title, 
                                            y_range=range(0, 101, 10),
                                            fontsize=38):
    
    global group_color_mapping
    
    # Create or retrieve the group colors
    unique_groups = sorted(df[groupby_col].unique())
    total_unique_groups = len(unique_groups)
    
    # If group_color_mapping is empty or doesn't have enough colors, initialize the color generator
    if len(group_color_mapping) < total_unique_groups:
        color_gen = color_generator(total_unique_groups)
    
        for group in unique_groups:
            if group not in group_color_mapping:
                # If the group doesn't have a color assigned yet, give it one
                group_color_mapping[group] = next(color_gen)

    # Set background color to 50% grey if the axis is empty
    if not ax.collections:
        ax.set_facecolor("#808080")  # This is the code for 50% grey
    
    # Set grid color to very light grey
    ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.5)
    
    # Set font color to very light grey for axes, title, and tick labels
    ax.tick_params(colors='#D3D3D3')
    ax.xaxis.label.set_color('#D3D3D3')
    ax.yaxis.label.set_color('#D3D3D3')
    ax.title.set_color('#D3D3D3')
    
    means = df[task_order+[groupby_col]].groupby(groupby_col).mean().T
    if show_sem:
        variation = df[task_order+[groupby_col]].groupby(groupby_col).sem().T
    elif show_std:
        variation = df[task_order+[groupby_col]].groupby(groupby_col).std().T
    else:
        variation = None
    
    # If "Show All" is toggled, display individual observations
    if show_all:
        for group in means.columns:
            group_data = df[df[groupby_col] == group][task_order].values
            for data in group_data:
                ax.plot(task_order, data, alpha=0.3, color=group_color_mapping[group], lw=1)
    
    # Plotting means with markers
    for group in means.columns:
        ax.plot(task_order, means[group], marker='o', lw=3, color=group_color_mapping[group], label=f"{group} - n={len(df[df[groupby_col] == group])}")
    
    # Adding shaded region for SEM or STD
    if variation is not None:
        for group in means.columns:
            ax.fill_between(task_order, 
                            means[group] - variation[group], 
                            means[group] + variation[group], 
                            color=group_color_mapping[group],
                            alpha=0.2)

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize-6)
    ax.set_ylabel(ylabel, fontsize=fontsize-6)
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels(task_order, rotation=90, fontsize=fontsize-10)
    ax.set_yticks(y_range)
    ax.set_yticklabels(y_range, fontsize=fontsize-10)
    ax.legend(title=legend_title, title_fontsize=fontsize-10, 
              fontsize=fontsize-10, loc='upper left', bbox_to_anchor=(1, 1))


def interactive_line_plot(df, groupby_col, FC_vars):
    unique_groups = sorted(df[groupby_col].unique())
    
    group_selector = widgets.SelectMultiple(
        options=unique_groups,
        value=list(unique_groups),
        description='',
        disabled=False
    )
    
    show_all_toggle = widgets.ToggleButton(
        value=False,
        description='Show All',
        disabled=False,
        button_style='', 
        tooltip='Display All Observations',
        layout=widgets.Layout(width='120px')
    )
    
    sem_toggle = widgets.ToggleButton(
        value=False,
        description='Show SEM',
        disabled=False,
        button_style='', 
        tooltip='Display Standard Error of Mean',
        layout=widgets.Layout(width='120px')
    )
    
    std_toggle = widgets.ToggleButton(
        value=False,
        description='Show STD',
        disabled=False,
        button_style='', 
        tooltip='Display Standard Deviation',
        layout=widgets.Layout(width='120px')
    )
    
    def update_plot(primary_selected_groups, secondary_selected_groups, show_sem, show_std, show_all):
        fig, ax = plt.subplots(figsize=(27, 15))
        
        # For primary groups
        primary_filtered_df = df[df[primary_groupby_col].isin(primary_selected_groups)]
        plot_avg_scores_by_group_with_variation(
            ax=ax,
            df=primary_filtered_df,
            task_order=FC_vars,
            groupby_col=primary_groupby_col,
            show_sem=show_sem,
            show_std=show_std,
            show_all=show_all,
            title="Mean values for selected groups",
            xlabel="Task",
            ylabel="Mean Value",
            legend_title=primary_groupby_col
        )
            
        # For secondary groups
        secondary_filtered_df = df[df[secondary_groupby_col].isin(secondary_selected_groups)]
        plot_avg_scores_by_group_with_variation(
            ax=ax,
            df=secondary_filtered_df,
            task_order=FC_vars,
            groupby_col=secondary_groupby_col,
            show_sem=show_sem,
            show_std=show_std,
            show_all=show_all,
            title="Mean values for selected groups",
            xlabel="Task",
            ylabel="Mean Value",
            legend_title=secondary_groupby_col
        )
        
        plt.tight_layout()
        plt.show()

    group_label = widgets.Label(value='Select Groups:')
    toggles = widgets.VBox([show_all_toggle, sem_toggle, std_toggle], layout=widgets.Layout(align_items='center'))
    left_box = widgets.VBox([group_label, group_selector])
    right_box = widgets.VBox([widgets.Label(value='Display Options:'), toggles])
    ui = widgets.HBox([left_box, right_box])
    
    out = widgets.interactive_output(update_plot, {
        'selected_groups': group_selector,
        'show_sem': sem_toggle,
        'show_std': std_toggle,
        'show_all': show_all_toggle
    })
    
    display(ui, out)


def dual_interactive_line_plot(df, primary_groupby_col, secondary_groupby_col, FC_vars):
    # Global variable to store the color mapping for each group.
    global group_color_mapping

    primary_unique_groups = sorted(df[primary_groupby_col].unique())
    secondary_unique_groups = sorted(df[secondary_groupby_col].unique())

    primary_group_selector = widgets.SelectMultiple(
        options=primary_unique_groups,
        value=list(primary_unique_groups),
        description='Primary Groups',
        disabled=False
    )

    secondary_group_selector = widgets.SelectMultiple(
        options=secondary_unique_groups,
        value=list(secondary_unique_groups),
        description='Secondary Groups',
        disabled=False
    )

    show_all_toggle = widgets.ToggleButton(
        value=False,
        description='Show All',
        disabled=False,
        button_style='', 
        tooltip='Display All Observations',
        layout=widgets.Layout(width='120px')
    )

    sem_toggle = widgets.ToggleButton(
        value=False,
        description='Show SEM',
        disabled=False,
        button_style='', 
        tooltip='Display Standard Error of Mean',
        layout=widgets.Layout(width='120px')
    )

    std_toggle = widgets.ToggleButton(
        value=False,
        description='Show STD',
        disabled=False,
        button_style='', 
        tooltip='Display Standard Deviation',
        layout=widgets.Layout(width='120px')
    )

    def update_plot(primary_selected_groups, secondary_selected_groups, show_sem, show_std, show_all):
        # For primary groups
        primary_filtered_df = df[df[primary_groupby_col].isin(primary_selected_groups)]
        plot_avg_scores_by_group_with_variation(
            df=primary_filtered_df,
            task_order=FC_vars,
            groupby_col=primary_groupby_col,
            show_sem=show_sem,
            show_std=show_std,
            show_all=show_all,
            title="Mean values for selected groups",
            xlabel="Task",
            ylabel="Mean Value",
            legend_title=primary_groupby_col
        )
        
        # For secondary groups
        secondary_filtered_df = df[df[secondary_groupby_col].isin(secondary_selected_groups)]
        plot_avg_scores_by_group_with_variation(
            df=secondary_filtered_df,
            task_order=FC_vars,
            groupby_col=secondary_groupby_col,
            show_sem=show_sem,
            show_std=show_std,
            show_all=show_all,
            title="Mean values for selected groups",
            xlabel="Task",
            ylabel="Mean Value",
            legend_title=secondary_groupby_col
        )

    ui = widgets.HBox([
        primary_group_selector, 
        secondary_group_selector, 
        show_all_toggle, 
        sem_toggle, 
        std_toggle
    ])
    
    out = widgets.interactive_output(update_plot, {
        'primary_selected_groups': primary_group_selector,
        'secondary_selected_groups': secondary_group_selector,
        'show_sem': sem_toggle,
        'show_std': std_toggle,
        'show_all': show_all_toggle
    })
    
    display(ui, out)


#%%
# Function to plot heatmap
def plot_heatmap(df, task_order, groupby_col, selected_group, figsize=(27, 15), fontsize=38):
    # Create a figure with custom size
    plt.figure(figsize=figsize)
    
    # Generate heatmap using seaborn without annotations in cells
    ax = sns.heatmap(df[task_order], cmap="viridis", linewidths=.5, cbar_kws={"shrink": 1, "label": "Score"})
    
    # Title with group name and number of subjects
    n_subjects = len(df)
    plt.title(f"{selected_group} (n={n_subjects})", fontsize=fontsize)
    
    # Adjust y-axis
    plt.ylabel("Individuals", fontsize=fontsize-6)
    ax.set_yticks([])  # Remove y-axis tickmarks
    plt.xlabel("Tasks", fontsize=fontsize-6)
    plt.yticks(fontsize=fontsize-10)
    plt.xticks(fontsize=fontsize-10, rotation=90)
    
    # Adjust colorbar font size and label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize-10)
    cbar.set_label("Score", size=fontsize-6)
    
    plt.tight_layout()

    # Display the plot
    plt.show()

# Interactive function
def interactive_heatmap(df, groupby_col, FC_vars):
    # Get unique group names
    group_names = sorted(df[groupby_col].unique())
    
    # Dropdown for group selection with custom font size
    group_selector = widgets.Dropdown(
        options=group_names,
        value=group_names[0],
        description='Select Group:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    # Interactive widget
    @widgets.interact(group=group_selector)
    def update_plot(group):
        filtered_df = df[df[groupby_col] == group]
        plot_heatmap(filtered_df, FC_vars, groupby_col, group)



#%%
# Function to plot radar/spider plot
def plot_radar(df, task_order, groupby_col, selected_group, figsize=(8, 8), fontsize=16):

    global group_color_mapping
    
    # Create or retrieve the group colors
    unique_groups = sorted(df[groupby_col].unique())
    total_unique_groups = len(unique_groups)
    
    # If group_color_mapping is empty, initialize the color generator
    if not group_color_mapping:
        color_gen = color_generator(total_unique_groups)
    
    for group in unique_groups:
        if group not in group_color_mapping:
            # If the group doesn't have a color assigned yet, give it one
            group_color_mapping[group] = next(color_gen)

    # Number of variables we're plotting
    num_vars = len(task_order)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Set figure and subplot size
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Set background color for the radar only
    ax.set_facecolor("#808080")
    
    # Helper function to plot data on radar chart
    def add_to_radar(data, color, label):
        values = data[task_order].tolist()
        values += values[:1]  # Complete the loop
        ax.plot(angles + angles[:1], values, color=color, linewidth=2, label=label)

    # Add each feature to the radar chart
    for idx, group in enumerate(df[groupby_col].unique()):
        # Select valid columns and calculate the mean to handle the FutureWarning
        valid_cols = df[df[groupby_col] == group][task_order]
        add_to_radar(valid_cols.mean(), group_color_mapping[group], f"{group} (n={len(valid_cols)})")
    
    # Set the angle, labels and location for each label
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    # Calculate label rotations: start from 90 and rotate continuously back to 90
    label_rotations = np.linspace(90, -270, num_vars).tolist()

    for angle, label, rotation in zip(angles, task_order, label_rotations):
        ha = 'center'
        ax.text(angle, ax.get_rmax() + 30, label, rotation=rotation, ha=ha, va='center', fontsize=fontsize-2, color='black')

    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.grid(color='lightgrey')
    
    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.7, 1.2), fontsize=fontsize-4)

    # Show plot
    plt.show()
    
def interactive_radar(df, groupby_col, FC_vars):
    # Get unique group names
    group_names = sorted(df[groupby_col].unique())
    
    # Dropdown for group selection
    group_selector = widgets.SelectMultiple(
        options=group_names,
        value=[group_names[0]],  # Default to the first group
        description='Select Group(s):',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    # Interactive widget
    @widgets.interact(group=group_selector)
    def update_plot(group):
        filtered_df = df[df[groupby_col].isin(group)]
        plot_radar(filtered_df, FC_vars, groupby_col, group)