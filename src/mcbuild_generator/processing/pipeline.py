from mcbuild_generator.processing.clean_data import clean_data
from mcbuild_generator.processing.transform_data import transform_data
from mcbuild_generator.utils.args import get_config


def pipeline(config):
    '''
    Data processing pipeline
    '''
    print('\ncleaning data...')
    clean_data(use_cache=config['use_cache'], multiproc=config['multiprocessing'])
    print('\ntransforming data...')
    transform_data()


if __name__=='__main__':
    # get args
    config = get_config()

    # run
    pipeline(config)