from local_datasets.data_EN.data_en import DataEN
from local_datasets.data_HU.data_hu import DataHU
from local_datasets.data_RO.data_ro import DataRO


if __name__ == "__main__":
    data_en = DataEN().get_data()
    data_hu = DataHU().get_data()
    data_ro = DataRO().get_data()
    
    print(data_en)
    print(data_hu)
    print(data_ro)