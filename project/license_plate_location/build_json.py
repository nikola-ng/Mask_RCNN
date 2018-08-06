import os

import json


class ImageLabels:
    def __init__(self, json_path):
        self.json_path = json_path
        if not os.path.exists(json_path):
            # create json if not exists
            open(json_path, 'w').close()
        else:
            self.json_file = open(json_path, 'rb')
            try:
                self.json_data = json.loads(self.json_file.read().decode('utf-8'))
            except json.decoder.JSONDecodeError:
                self.json_data = {}

    def add_serial(self, filename='', size=0,
                   region_type='polygon', region_points=[],
                   file_attributes={}):
        # define key
        serial_name = filename + str(size)
        try:
            key = self.json_data[serial_name]
        except KeyError:
            serial = {
                # 'filename': filename.encode('unicode_escape'),  # translate Chinese characters into unicode
                'filename': filename,
                'size': size,
                'regions': [],
                'file_attributes': file_attributes
            }
            # update json_data
            self.json_data[serial_name] = serial

        # if serial exists, append region directly
        self.add_regions(serial_name, region_type, region_points)

    def add_regions(self, serial_name, region_type, region_points):
        region = {'shape_attributes': {},
                  'region_attributes': {}}
        region['shape_attributes']['name'] = region_type
        region['shape_attributes']['all_points_x'] = [p[0] for p in region_points]
        region['shape_attributes']['all_points_y'] = [p[1] for p in region_points]

        self.json_data[serial_name]['regions'].append(region)

    def update(self):
        self.json_data = json.dumps(self.json_data, ensure_ascii=False)  # dict to json string
        self.json_file = open(self.json_path, 'wb')
        self.json_file.write(self.json_data.encode())

    def close(self):
        self.json_file.close()


if __name__ == '__main__':
    # json_path = '../../datasets/license_plate/raw/via_region_data.json_data'
    # js = json_data.loads(open(json_path).read())
    # print()
    il = ImageLabels('./j.json')
    il.add_serial('广州', 123456, 'polygon', [[1, 1], [2, 2], [3, 4], [45, 99]])
    il.add_serial('广州', 123456, 'polygon', [[333, 333], [2333, 2333], [31, 41], [415, 919]])
    il.add_serial('北京', 22)
    il.update()
    il.close()
