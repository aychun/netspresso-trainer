from netspresso_trainer import train_cli


def train_with_inline_yaml():
    from netspresso_trainer import train_with_yaml
    logging_dir = train_with_yaml(
        # gpus="0,1",
        data="config/data/traffic-sign.yaml",
        # data="config/data/beans.yaml",
        augmentation="config/augmentation/detection.yaml",
        model="config/model/yolo-fastest/yolo-fastest.yaml",
        training="config/training.yaml",
        logging="config/logging.yaml",
        environment="config/environment.yaml",
        log_level='INFO'
    )
    return logging_dir


if __name__ == '__main__':
    # logging_dir = train_cli()

    # With inline yaml
    logging_dir = train_with_inline_yaml()
    
    print(f"Training results are saved at: {logging_dir}")