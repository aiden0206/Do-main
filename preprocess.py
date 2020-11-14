import numpy as np
class preprocess:
    def __init__(self, file_path, num_channels):
        self._file = file_path
        self._channels = num_channels
    
    # collected from ADS 1299 
    def file_load(self):
        self.data = {}
        for i in range(self._channels):
            file = open(self._file)
            _list = []
            while True:
                line = file.readline()
                if not line: break
                _list.append(float(line.split('\t')[i]))
            self.data['channel'+str(i)] = _list
        file.close()
        
        #print('self.data 변수에 dictionary 형식으로 저장.')
        return 
    
    # base_range: [start, end]의 list. 0점으로 맞출 데이터의 범위이며 해당 구간 데이터의 평균을 0점으로 간주.
    def normalize(self, base_range):
        self.norm = {}
        for i in range(self._channels):
            file = open(self._file)
            _list = []
            while True:
                line = file.readline()
                if not line: break
                _list.append(float(line.split('\t')[i]))
            self.norm['channel'+str(i)] = _list
        file.close()
        
        for i in range(self._channels):
            _sum = 0
            start, end = base_range[0], base_range[1]
            key = 'channel' + str(i)
            for j in self.norm[key][start:end]:
                _sum += j
            avg = _sum / (end - start)

            norm = []
            for j in self.norm[key]:
                norm.append(j - avg)
            _max = np.array(norm).max()
            _min = -np.array(norm).min()
            
            if _max > _min:
                self.norm[key] = norm / _max
            else:
                self.norm[key] = norm / _min
        
        #print('self.norm 변수에 dictionary 형식으로 저장.')
        return
    
    # data: 1D array
    def linear_baseline(self, data, start_range=[5, 60], end_range=[950, 980]):
        start, end = start_range[0], start_range[1]
        mid = (end + start) / 2
        _sum = 0
        for i in range(start, end):
            _sum += data[i]
        avg = _sum / (end - start)
        
        start, end = end_range[0], end_range[1]
        mid1 = (end + start) / 2
        _sum = 0
        for i in range(start, end):
            _sum += data[i]
        avg1 = _sum / (end - start)
        
        a = (avg1 - avg) / (mid1 - mid)
        data_correction = []
        for i in range(len(data)):
            data_correction.append(data[i] - a * (i - mid))
        
        return data_correction