


from src.scripts.convert_dataset import Convertor


if __name__ == '__main__':
    source = '/data/tml/mixed_polyp'
    target = '/data/tml/mixed_polyp_v5_format2'
    convertor = Convertor(source_path=source, target_path=target)
    convertor()

