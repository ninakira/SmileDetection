from data_access import ImageDaoKeras

def main():
    print("Hey you! Smile!")
    dao_single_path = ImageDaoKeras(data_path="images/train")
    # dao_separate_paths = ImageDaoKeras(train_path="images/train", validation_path="images/test")

if __name__ == "__main__":
    main()