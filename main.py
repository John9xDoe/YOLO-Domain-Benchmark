import synth_data_generator
from tqdm import tqdm

if __name__ == '__main__':
    import torch

    print("CUDA доступна:", torch.cuda.is_available())
    print("Количество GPU:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Название GPU:", torch.cuda.get_device_name(0))
        print("Память GPU:", torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, "GB")

    '''
    from ultralytics.data.utils import check_det_dataset
    print(check_det_dataset('synth_data.yml'))
    '''

    '''
    n = int(input('Enter number of images: '))
    for i in tqdm(range(n)):
        synth_data_generator.generate_object(save_path='data', filename=i, part='val')
    '''