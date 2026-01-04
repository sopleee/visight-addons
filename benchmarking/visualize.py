import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

def line_plot_grouped(df: pd.DataFrame, output_dir: str, 
                    x_axis:str, x_name:str, grouping_col:str, grouping_name:str, 
                    y_axis:str="miss_rate", y_name:str="Miss rate",
                    evenly_space_x_axis=False):
    
    # Ensure numeric type only for y_axis
    df[y_axis] = pd.to_numeric(df[y_axis])
    
    # Try to convert x_axis to numeric if possible, otherwise keep as-is
    try:
        df[x_axis] = pd.to_numeric(df[x_axis])
        x_is_numeric = True
    except (ValueError, TypeError):
        x_is_numeric = False

    # Sort data
    df = df.sort_values(by=[grouping_col, x_axis])

    # Unique sorted values
    grouping_vals = sorted(df[grouping_col].unique())
    
    # Get unique x values (sorted if numeric)
    if x_is_numeric:
        x_values = sorted(df[x_axis].unique())
    else:
        # For non-numeric, preserve order from dataframe
        x_values = df[x_axis].unique()

    # Create position indices for even spacing
    x_positions = list(range(len(x_values)))
    x_map = dict(zip(x_values, x_positions))

    # Marker styles
    markers = ['s', 'o', '^', 'D', 'v', 'x', '*', 'P', '>']

    plt.figure(figsize=(9, 5))

    for i, gv in enumerate(grouping_vals):
        data = df[df[grouping_col] == gv].copy()
        # Map x values to positions
        data['x_pos'] = data[x_axis].map(x_map)

        x_plot = data[x_axis] if (not evenly_space_x_axis and x_is_numeric) else data['x_pos']
        
        # Plot the total line
        plt.plot(
            x_plot,
            data[y_axis],
            marker=markers[i % len(markers)],
            label=gv,
            linewidth=1.5,
            color=f'C{i}'
        )
        
        # Add stacked area regions
        if 'inf_sec_per_frame' in df.columns and 'noninf_sec_per_frame' in df.columns:
            # First layer: non-inference (from 0 to noninf_sec_per_frame)
            plt.fill_between(
                x_plot,
                0,
                data['noninf_sec_per_frame'],
                alpha=0.3,
                color=f'C{i}',
                label=f'{gv} - Non-Inference'
            )
            
            # Second layer: inference (from noninf_sec_per_frame to sec_per_frame)
            plt.fill_between(
                x_plot,
                data['noninf_sec_per_frame'],
                data[y_axis],
                alpha=0.5,
                color=f'C{i}',
                label=f'{gv} - Inference'
            )

    # X-axis: for non-numeric or evenly_space_x_axis, use positions with labels
    if evenly_space_x_axis or not x_is_numeric:
        plt.xticks(ticks=x_positions, labels=[str(x) for x in x_values])
    
    plt.xlabel(x_name)

    # Y-axis: dynamic scaling
    y_max = df[y_axis].max()
    y_min = df[y_axis].min()
    y_tick_max = 0.5 #round((y_max + 0.001) * 10) / 10
    y_tick_min = 0
    y_avg = (y_tick_max - y_tick_min) / len(df[y_axis].unique()) * 0.8
    plt.ylim(max(0, y_tick_min - y_avg), y_tick_max + y_avg)

    plt.yticks(
        ticks=[x for x in plt.yticks()[0] if x <= y_tick_max],
        labels=[round(max(y, 0), 2) for y in plt.yticks()[0] if y <= y_tick_max],
    )

    # Aesthetics
    plt.ylabel(y_name)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title=grouping_name, loc='upper right')
    plt.tight_layout()
    
    # Save before show (or figure won't be saved properly)
    plt.savefig(output_dir)

def line_plot_grouped2(df: pd.DataFrame, output_dir: str, 
                    x_axis:str, x_name:str, grouping_col:str, grouping_name:str, 
                    y_axis:str="miss_rate", y_name:str="Miss rate",
                    evenly_space_x_axis=False):
    
    # Ensure numeric type only for y_axis
    df[y_axis] = pd.to_numeric(df[y_axis])
    
    # Try to convert x_axis to numeric if possible, otherwise keep as-is
    try:
        df[x_axis] = pd.to_numeric(df[x_axis])
        x_is_numeric = True
    except (ValueError, TypeError):
        x_is_numeric = False

    # Sort data
    df = df.sort_values(by=[grouping_col, x_axis])

    # Unique sorted values
    grouping_vals = sorted(df[grouping_col].unique())
    
    # Get unique x values (sorted if numeric)
    if x_is_numeric:
        x_values = sorted(df[x_axis].unique())
    else:
        # For non-numeric, preserve order from dataframe
        x_values = df[x_axis].unique()

    # Create position indices for even spacing
    x_positions = list(range(len(x_values)))
    x_map = dict(zip(x_values, x_positions))

    # Marker styles
    markers = ['s', 'o', '^', 'D', 'v', 'x', '*', 'P', '>']

    plt.figure(figsize=(9, 5))

    for i, gv in enumerate(grouping_vals):
        data = df[df[grouping_col] == gv].copy()
        # Map x values to positions
        data['x_pos'] = data[x_axis].map(x_map)

        x_plot = data[x_axis] if (not evenly_space_x_axis and x_is_numeric) else data['x_pos']
        
        plt.plot(
            x_plot,
            data[y_axis],
            marker=markers[i % len(markers)],
            label=gv,
            linewidth=1.5
        )
        
        # Add shaded region between inference and non-inference time
        if 'inf_sec_per_frame' in df.columns and 'noninf_sec_per_frame' in df.columns:
            plt.fill_between(
                x_plot,
                data['inf_sec_per_frame'],
                data['noninf_sec_per_frame'],
                alpha=0.2
            )

    # X-axis: for non-numeric or evenly_space_x_axis, use positions with labels
    if evenly_space_x_axis or not x_is_numeric:
        plt.xticks(ticks=x_positions, labels=[str(x) for x in x_values])
    
    plt.xlabel(x_name)

    # Y-axis: dynamic scaling
    y_max = df[y_axis].max()
    y_min = df[y_axis].min()
    y_tick_max = round((y_max + 0.001) * 10) / 10
    y_tick_min = 0
    y_avg = (y_tick_max - y_tick_min) / len(df[y_axis].unique()) * 0.8
    plt.ylim(max(0, y_tick_min - y_avg), y_tick_max + y_avg)

    plt.yticks(
        ticks=[x for x in plt.yticks()[0] if x <= y_tick_max],
        labels=[round(max(y, 0), 2) for y in plt.yticks()[0] if y <= y_tick_max],
    )

    # Aesthetics
    plt.ylabel(y_name)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title=grouping_name, loc='upper right')
    plt.tight_layout()
    
    # Save before show (or figure won't be saved properly)
    plt.savefig(output_dir)

def line_plot_grouped1(df: pd.DataFrame, output_dir: str, 
                    x_axis:str, x_name:str, grouping_col:str, grouping_name:str, 
                    y_axis:str="miss_rate", y_name:str="Miss rate",
                    evenly_space_x_axis=False):
    
    # Ensure numeric type only for y_axis
    df[y_axis] = pd.to_numeric(df[y_axis])
    
    # Try to convert x_axis to numeric if possible, otherwise keep as-is
    try:
        df[x_axis] = pd.to_numeric(df[x_axis])
        x_is_numeric = True
    except (ValueError, TypeError):
        x_is_numeric = False

    # Sort data
    df = df.sort_values(by=[grouping_col, x_axis])

    # Unique sorted values
    grouping_vals = sorted(df[grouping_col].unique())
    
    # Get unique x values (sorted if numeric)
    if x_is_numeric:
        x_values = sorted(df[x_axis].unique())
    else:
        # For non-numeric, preserve order from dataframe
        x_values = df[x_axis].unique()

    # Create position indices for even spacing
    x_positions = list(range(len(x_values)))
    x_map = dict(zip(x_values, x_positions))

    # Marker styles
    markers = ['s', 'o', '^', 'D', 'v', 'x', '*', 'P', '>']

    plt.figure(figsize=(9, 5))

    for i, gv in enumerate(grouping_vals):
        data = df[df[grouping_col] == gv].copy()
        # Map x values to positions
        data['x_pos'] = data[x_axis].map(x_map)

        plt.plot(
            data[x_axis] if (not evenly_space_x_axis and x_is_numeric) else data['x_pos'],
            data[y_axis],
            marker=markers[i % len(markers)],
            label=gv,
            linewidth=1.5
        )

    # X-axis: for non-numeric or evenly_space_x_axis, use positions with labels
    if evenly_space_x_axis or not x_is_numeric:
        plt.xticks(ticks=x_positions, labels=[str(x) for x in x_values])
    
    plt.xlabel(x_name)

    # Y-axis: dynamic scaling
    y_max = df[y_axis].max()
    y_min = df[y_axis].min()
    y_tick_max = round((y_max + 0.001) * 10) / 10 #+ 0.02
    y_tick_min = 0 #round((y_min - 0.01) * 10) / 10
    y_avg = (y_tick_max - y_tick_min) / len(df[y_axis].unique()) * 0.8
    plt.ylim(max(0, y_tick_min - y_avg), y_tick_max + y_avg)

    plt.yticks(
        ticks=[x for x in plt.yticks()[0] if x <= y_tick_max],
        labels=[round(max(y, 0), 2) for y in plt.yticks()[0] if y <= y_tick_max],  # Round to 2 decimals
    )

    # Aesthetics
    plt.ylabel(y_name)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(title=grouping_name, loc='upper right')
    plt.tight_layout()
    
    # Save before show (or figure won't be saved properly)
    plt.savefig(output_dir)
    # plt.show()

def violins(df, title, outpath): 
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plot_data = pd.melt(
        df, value_vars=['fetch rate', 'issue rate', 'execute rate', "ipc"],
        var_name='Metric', value_name='Rate')
    sns.violinplot(x='Metric', y='Rate', data=plot_data, palette='pastel',  # Try: 'Set2', 'pastel', 'muted', 'husl', 'colorblind'
                inner='quartile',cut=0, linewidth=1.5)
    plt.title(title)
    plt.savefig(outpath)
    plt.close()

def geometric_avg(df, group_cols, target_cols):    
    agg_dict = {col: [np.prod, 'count'] for col in target_cols}
    
    geom_avg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Calculate geometric means
    for col in target_cols:
        geom_avg_df[f"g_avg_{col}"] = (
            geom_avg_df[(col, 'prod')] ** (1 / geom_avg_df[(col, 'count')])
        )
        geom_avg_df = geom_avg_df.drop([(col, 'prod'), (col, 'count')], axis=1)
    
    geom_avg_df.columns = [col[0] if isinstance(col, tuple) else col 
                            for col in geom_avg_df.columns]
    return geom_avg_df

if __name__ == "__main__":
    ### AI CREDIT
    '''
    AI wrote line_plot_grouped and I've since tweaked it.
    '''
    filepath = f"./res.csv"
    df = pd.read_csv(filepath)
    df["inf_sec_per_frame"] = df["inference_time"]/df["frames"]
    df["sec_per_frame"] = df["total_time"]/df["frames"]
    df["noninf_sec_per_frame"] = (df["total_time"] - df["inference_time"]) /df["frames"]
    res = df.groupby(["GPU", "frames"]).mean().reset_index()
    print(res)
    # line_plot_grouped(res, "./inference_speed.jpeg", "frames", "num_frames", 
    #                   "GPU", "GPU", "inf_sec_per_frame", "Frames Per Second (inference)")
    # line_plot_grouped(res, "./total_speed.jpeg", "frames", "num_frames", 
    #                   "GPU", "GPU", "sec_per_frame", "Frames Per Second (total)")
    # line_plot_grouped(res, "./non_inference_speed.jpeg", "frames", "num_frames", 
    #                   "GPU", "GPU", "noninf_sec_per_frame", "Frames Per Second (total non-inference)")
    
    # line_plot_grouped(
    #     df=res,  # your dataframe
    #     output_dir='frames_per_second_plot.png',  # or whatever filename you want
    #     x_axis='frames',  # or 'num_frames' depending on your column name
    #     x_name='num_frames',
    #     grouping_col='GPU',
    #     grouping_name='GPU',
    #     y_axis='sec_per_frame',
    #     y_name='Frames Per Second (total)',
    #     evenly_space_x_axis=False
    # )
    line_plot_grouped(
        df=res[res['GPU'] == 'A10'],
        output_dir='a10_frames_per_second.png',
        x_axis='frames',
        x_name='num_frames',
        grouping_col='GPU',
        grouping_name='Grouping',
        y_axis='sec_per_frame',
        y_name='Frames Per Second (total)',
        evenly_space_x_axis=False
    )
    
    line_plot_grouped(
        df=res[res['GPU'] == 'T4'],
        output_dir='t4_frames_per_second.png',
        x_axis='frames',
        x_name='num_frames',
        grouping_col='GPU',
        grouping_name='Grouping',
        y_axis='sec_per_frame',
        y_name='Frames Per Second (total)',
        evenly_space_x_axis=False
    )

