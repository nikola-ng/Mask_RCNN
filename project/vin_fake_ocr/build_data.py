import os
import pandas
import shutil
import numpy as np

raw_images_path = '/home/public/car_exam/raw_images'
output_path = '/home/wang/vin'
ipl_par = os.listdir(raw_images_path)
GROUND_TRUTH_CSV = './ground_truth.csv'

df = pandas.read_csv(GROUND_TRUTH_CSV, encoding='utf-8')
no = np.asarray(df['作业流水号']).astype(str)
vin_no = np.asarray(df['车架号']).astype(str)

lp_dict = {}
for i, n in enumerate(no):
    lp_dict[n] = vin_no[i]

for ip_par in ipl_par:
    if os.path.isdir(raw_images_path + '/' + ip_par):
        for ip in os.listdir(raw_images_path + '/' + ip_par):
            if ip.endswith('.jpg') and '0113' in ip:
                print(ip_par)
                try:
                    shutil.copy(raw_images_path + '/' + ip_par + '/' + ip,
                            output_path + '/{}_{}_'.format(lp_dict[ip_par], ip_par) + ip)

                except KeyError:
                    shutil.copy(raw_images_path + '/' + ip_par + '/' + ip,
                                output_path + '/{}_{}_'.format('UNKNOWN', ip_par) + ip)
                except FileNotFoundError:
                    pass