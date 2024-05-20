import os


def list_files(data_path, dataset_format, sets, semantic_feat):
    sets_list = sets.split("-")

    image_path_list = []
    semantic_map_path_list = []

    if dataset_format == "cityscapes":
        image_dir = "leftImg8bit"
        semantic_dir = "gtFine"

        image_path = os.path.join(data_path, "leftImg8bit")
        semantic_path = os.path.join(data_path, "gtFine")

        for set in os.listdir(image_path):
            if os.path.isdir(os.path.join(image_path,
                                          set)) and set in sets_list:
                image_set_path = os.path.join(image_path, set)
                semantic_set_path = os.path.join(semantic_path, set)
                for city in os.listdir(image_set_path):
                    image_city_path = os.path.join(image_set_path, city)
                    semantic_city_path = os.path.join(semantic_set_path, city)
                    for file in os.listdir(image_city_path):
                        file_basename = "_".join(file.split("_")[:3])
                        if city in file_basename:
                            image_path_list.append(
                                os.path.join(image_city_path, file))
                            if semantic_feat:
                                semantic_map_path_list.append(
                                    os.path.join(
                                        semantic_city_path, "_".join([
                                            file_basename, "gtFine", "labelIds"
                                        ]) + '.png'))

    elif dataset_format == "mapillary":
        image_dir = "images"
        semantic_dir = "instances"

        for set in os.listdir(data_path):
            set_path = os.path.join(data_path, set)
            if os.path.isdir(set_path) and set in sets_list:
                images_path = os.path.join(set_path, image_dir)
                semantic_path = os.path.join(set_path, semantic_dir)
                for file in os.listdir(images_path):
                    image_path_list.append(os.path.join(images_path, file))
                    if semantic_feat:
                        semantic_map_path_list.append(
                            os.path.join(semantic_path,
                                         file.replace('.jpg', '.png')))


    else:
        raise Exception("Invalid dataset format.")

    return image_path_list, semantic_map_path_list
