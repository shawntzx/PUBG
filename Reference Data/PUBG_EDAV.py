import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

stat = pd.read_csv('kill_match_stats_final_2.csv')
kill_stat = stat[stat['map'] == 'ERANGEL']
stat = pd.read_csv('kill_match_stats_final_3.csv')
stat = stat[stat['map'] == 'ERANGEL']
kill_stat = pd.concat([kill_stat, stat])
stat = pd.read_csv('kill_match_stats_final_4.csv')
stat = stat[stat['map'] == 'ERANGEL']
kill_stat = pd.concat([kill_stat, stat]).reset_index(drop=True)

stat = pd.read_csv('agg_match_stats_2.csv')
match_stat = stat[stat['party_size'] == 1]
stat = pd.read_csv('agg_match_stats_3.csv')
stat = stat[stat['party_size'] == 1]
match_stat = pd.concat([match_stat, stat])
stat = pd.read_csv('agg_match_stats_4.csv')
stat = stat[stat['party_size'] == 1]
match_stat = pd.concat([match_stat, stat]).reset_index(drop=True)

match_set = set(match_stat['match_id'])

kill_1_erang = kill_stat[kill_stat['match_id'].isin(match_set)].reset_index(drop=True)

kill_1_erang = kill_1_erang.sort_values(by=['match_id', 'time']).reset_index(drop=True)
kill_1_erang['total_dead'] = kill_1_erang.groupby(['match_id']).cumcount()+1

dead_time = kill_1_erang.groupby('total_dead')['time'].mean().reset_index(name='mean')

plt.figure(figsize=(9, 5))
plt.scatter(x=dead_time['mean'], y=dead_time['total_dead'], color='r', s=4)
plt.xlabel('average time')
plt.ylabel('total dead')
plt.title('Scatter plot of mean time to reach total dead')
plt.show()

plt.figure(figsize=(9, 5))
plt.scatter(x=dead_time['mean'], y=dead_time['total_dead'], color='r', s=4)
plt.xscale('log')
plt.xlabel('log(average time)')
plt.ylabel('total dead')
plt.title('Scatter plot of log mean time to reach total dead')
plt.show()

# Remove outliers

kill_1_erang = kill_1_erang[(kill_1_erang['killer_position_x'] >= 0) &
                            (800000 >= kill_1_erang['killer_position_x']) &
                            (kill_1_erang['killer_position_y'] >= 0) &
                            (800000 >= kill_1_erang['killer_position_y']) &
                            (kill_1_erang['victim_position_x'] >= 0) &
                            (800000 >= kill_1_erang['victim_position_x']) &
                            (kill_1_erang['victim_position_y'] >= 0) &
                            (800000 >= kill_1_erang['victim_position_y'])].reset_index(drop=True)

kill_1_erang['killer_position_x'] = kill_1_erang['killer_position_x'].values*4096/800000
kill_1_erang['killer_position_y'] = kill_1_erang['killer_position_y'].values*4096/800000

kill_1_erang['victim_position_x'] = kill_1_erang['victim_position_x'].values*4096/800000
kill_1_erang['victim_position_y'] = kill_1_erang['victim_position_y'].values*4096/800000

x_scaled = kill_1_erang['killer_position_x'].dropna()
y_scaled = kill_1_erang['killer_position_y'].dropna()

heatmap, xedges, yedges = np.histogram2d(x_scaled, y_scaled, bins=100)
heatmap = gaussian_filter(heatmap, sigma=1.5)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
alphas = np.clip(Normalize(0, heatmap.max(), clip=True)(heatmap)*10, 0, 1)
colors = Normalize(0, heatmap.max(), clip=True)(heatmap)
colors = cm.Reds(colors)
colors[..., -1] = alphas

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 4096)
ax.set_ylim(0, 4096)
ax.imshow(imread('erangel.jpg'))
ax.imshow(colors, extent=extent, origin='lower', cmap=cm.Reds, alpha=0.9)
ax.set_title('Heatmap of Killer Position')
plt.gca().invert_yaxis()

pivot_class = kill_1_erang.groupby(by=['killed_by']).size().reset_index(name='counts')
pivot_class = pivot_class.sort_values('counts').reset_index(drop=True)
pivot_high = pivot_class[pivot_class['counts'] >= 25000]
pivot_low = pivot_class[pivot_class['counts'] < 25000]

fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(12, 7))
ax1.scatter(x=pivot_high['counts'], y=pivot_high['killed_by'], color='r', s=4)
ax1.set_title('Cleveland dot plot of kill by method (count >= 25,000)', size=10)
ax1.tick_params(axis="y", labelsize=8)
ax1.set_xlabel('count', size=9)

ax2.scatter(x=pivot_low['counts'], y=pivot_low['killed_by'], color='r', s=4)
ax2.set_title('Cleveland dot plot of kill by method (count < 25,000)', size=10)
ax2.tick_params(axis="y", labelsize=8)
ax2.set_xlabel('count', size=9)
fig.tight_layout()
plt.show()

times = [0, 250, 500, 1000, 2180]
for i in range(4):
    circle = kill_1_erang[(kill_1_erang['time'] >= times[i]) & (kill_1_erang['time'] < times[i+1])]
    pivot_class = circle.groupby(by=['killed_by']).size().reset_index(name='counts')
    pivot_class = pivot_class.sort_values('counts').reset_index(drop=True)
    plt.figure(figsize=(12, 7))
    plt.scatter(x=pivot_class['counts'], y=pivot_class['killed_by'], color='r', s=4)
    title = 'Cleveland dot plot of kill by method from ' + str(times[i]) + '-' + str(times[i+1]) + ' seconds'
    plt.xlabel('count', size=9)
    plt.yticks(fontsize=8)
    plt.title(title, size=10)
    plt.show()

kill_first = kill_1_erang[kill_1_erang['killer_placement'] == 1].reset_index(drop=True)
kill_first = kill_first.sort_values(['match_id', 'time']).reset_index(drop=True)
kill_first = kill_first.groupby('match_id').tail(10).reset_index(drop=True)

kill_first['total_kill'] = kill_first.groupby(['match_id']).cumcount()+1

kill_first.to_csv('kill_first.csv', index=False)

# Get alluvial diagram in RMD

match_set = set(kill_1_erang['match_id'])
match_1_erang = match_stat[match_stat['match_id'].isin(match_set)].reset_index(drop=True)
pivot_kills = match_1_erang.groupby(by=['player_name'])['player_kills'].sum().reset_index(name='kill_counts')

match_1_erang['lose'] = [0 if i == 1 else 1 for i in match_1_erang['team_placement']]
pivot_deaths = match_1_erang.groupby(by=['player_name'])['lose'].sum().reset_index(name='death_counts')

death_dict = pd.Series(pivot_deaths['death_counts'].values, index=pivot_deaths['player_name']).to_dict()

pivot_kills['death_counts'] = 0
for i in range(len(pivot_kills)):
    if pivot_kills['player_name'][i] in death_dict:
        pivot_kills.at[i, 'death_counts'] = death_dict[pivot_kills['player_name'][i]]

pivot_kills['K/D'] = pivot_kills['kill_counts']/pivot_kills['death_counts']

# Remove outliers
pivot_kills = pivot_kills[pivot_kills['death_counts'] >= 15].reset_index(drop=True)

pivot_kills = pivot_kills.sort_values(['K/D', 'kill_counts'], ascending=[False, False]).reset_index(drop=True)

plt.figure(figsize=(9, 7))
plt.hist(pivot_kills['K/D'], bins=50)
plt.xlabel('K/D', size=9)
plt.ylabel('count', size=9)
plt.title('Histogram of Kill-Death ratio', size=10)
plt.show()

pivot_high = pivot_kills[pivot_kills['K/D'] > 3.0]
pivot_low = pivot_kills[pivot_kills['K/D'] <= 3.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
ax1.hist(pivot_low['K/D'], bins=20)
ax1.set_title('Histogram of Kill-Death ratio (<= 3.0)', size=10)
ax1.set_xlabel('K/D', size=9)
ax1.set_ylabel('count', size=9)

ax2.hist(pivot_high['K/D'], bins=20)
ax2.set_title('Histogram of Kill-Death ratio (> 3.0)', size=10)
ax2.set_xlabel('K/D', size=9)
ax2.set_ylabel('count', size=9)
fig.tight_layout()
plt.show()

sub_group = pivot_kills[(pivot_kills['death_counts'] > 45) & (pivot_kills['death_counts'] <= 50) &
                        (pivot_kills['kill_counts'] >= 40)]

top3 = sub_group.head(3)['player_name'].values.tolist()
for i in range(3):
    player = kill_1_erang[((kill_1_erang['killer_name'] == top3[i]) &
                           (kill_1_erang['killer_position_x'] > 0)) |
                          ((kill_1_erang['victim_name'] == top3[i]) &
                           (kill_1_erang['victim_position_x'] > 0))].reset_index(drop=True)
    player['match_id'] = player['match_id'].astype('category').cat.codes
    color = cm.hsv(np.linspace(0, 1, len(player['match_id'].unique())))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 4096)
    ax.set_ylim(0, 4096)
    ax.imshow(imread('erangel.jpg'), alpha=0.9)

    groups = player.groupby('match_id')
    for name, group in groups:
        x_last = group.tail(1)['killer_position_x'].values
        y_last = group.tail(1)['killer_position_y'].values
        if group.tail(1)['victim_name'].values[0] == top3[i]:
            x_last = group.tail(1)['victim_position_x'].values[0]
            y_last = group.tail(1)['victim_position_y'].values[0]
            group = group[['killer_position_x', 'killer_position_y']]
            group = group.append({'killer_position_x': x_last,
                                  'killer_position_y': y_last}, ignore_index=True)
        x = group['killer_position_x'].values
        y = group['killer_position_y'].values
        for j in range(len(x) - 1):
            ax.plot([x[j], x[j+1]], [y[j], y[j+1]], c=color[name].reshape(1,-1), linewidth=1)
        ax.scatter(x, y, s=6, c=color[name].reshape(1,-1))
        ax.scatter(x_last, y_last, s=18, marker='^', c=color[name].reshape(1,-1))
    ax.set_title('Approximate path of top player ' + str(i+1) + ' in each game')
    plt.gca().invert_yaxis()

low3 = sub_group.tail(3)['player_name'].values.tolist()
for i in range(3):
    player = kill_1_erang[((kill_1_erang['killer_name'] == low3[i]) &
                           (kill_1_erang['killer_position_x'] > 0)) |
                          ((kill_1_erang['victim_name'] == low3[i]) &
                           (kill_1_erang['victim_position_x'] > 0))].reset_index(drop=True)
    player['match_id'] = player['match_id'].astype('category').cat.codes
    color = cm.hsv(np.linspace(0, 1, len(player['match_id'].unique())))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 4096)
    ax.set_ylim(0, 4096)
    ax.imshow(imread('erangel.jpg'), alpha=0.9)

    groups = player.groupby('match_id')
    for name, group in groups:
        x_last = group.tail(1)['killer_position_x'].values
        y_last = group.tail(1)['killer_position_y'].values
        if group.tail(1)['victim_name'].values[0] == low3[i]:
            x_last = group.tail(1)['victim_position_x'].values[0]
            y_last = group.tail(1)['victim_position_y'].values[0]
            group = group[['killer_position_x', 'killer_position_y']]
            group = group.append({'killer_position_x': x_last,
                                  'killer_position_y': y_last}, ignore_index=True)
        x = group['killer_position_x'].values
        y = group['killer_position_y'].values
        for j in range(len(x) - 1):
            ax.plot([x[j], x[j+1]], [y[j], y[j+1]], c=color[name].reshape(1,-1), linewidth=1)
        ax.scatter(x, y, s=6, c=color[name].reshape(1,-1))
        ax.scatter(x_last, y_last, s=18, marker='^', c=color[name].reshape(1,-1))
    ax.set_title('Approximate path of lower player ' + str(i+1) + ' in each game')
    plt.gca().invert_yaxis()

# Remove outliers
match_stat = match_1_erang[match_1_erang['player_survive_time'] <= 3000].reset_index(drop=True)
match_stat = match_stat[match_stat['player_kills'] <= 30].reset_index(drop=True)

match_stat['total_dist'] = match_stat['player_dist_ride'] + match_stat['player_dist_walk']

match_stat['kill_ratio'] = match_stat['player_kills']/match_stat['total_dist']

# Remove outliers
match_stat = match_stat[match_stat['kill_ratio'] < 3.0].reset_index(drop=True)

plt.figure(figsize=(9, 7))
plt.scatter(x=match_stat['kill_ratio'], y=match_stat['team_placement'], s=4, alpha=0.2)
plt.xlabel('kill ratio', size=9)
plt.ylabel('team placement', size=9)
plt.title('Scatter plot of team placement vs kill ratio', size=10)
plt.show()