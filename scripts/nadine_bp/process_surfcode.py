import numpy as np
import os
import glob
import pandas as pd
import re

def round_to_higher_odd(number):
    rounded_number = round(number)
    if rounded_number % 2 == 0:
        return rounded_number + 1
    else:
        return rounded_number

# d = 5
n = 2025
d_s = 5
N = n * d_s ** 2
k = n / 20
d_baseline = round_to_higher_odd(np.ceil(np.sqrt(N / k)))
T_H = 10
r = 10
syndmeas_official = int(3 * np.sqrt(n) * d_s * T_H / r)
print(d_baseline)

def
# agg_df['p_bar'] = agg_df.apply(lambda row: 1 - (1 - 2 * row['p_fail']) ** (1 / 200), axis=1)
# agg_df['p_fail_real675'] = agg_df.apply(lambda row: 1/2*(1 - (1 - row['p_bar']) ** (syndmeas_official)), axis=1)
agg_df['p_fail_real675'] = 1 / 2 * (1 - (1 - 2 * agg_df['p_fail']) ** (
            675 / 200))  # agg_df.apply(lambda row: 1/2*(1 - (1 - row['p_bar']) ** (syndmeas_official)), axis=1)

agg_df['total_p_fail'] = 1 - (1 - agg_df['p_fail_real675']) ** k

new_df = agg_df[agg_df['syndmeas'] == 200]

filtered_df = new_df  # .query('total_p_fail != 0 and not total_p_fail.isnull() and num_samples >= 20')

filtered_df = filtered_df.sort_values(by='p')
filtered_df.head(20)