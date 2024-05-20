import torch


def check_present_classes(semantics,
                          classes_in_questions,
                          label_name_info_inv,
                          min_ratio=0.01):
    tot_px_count = semantics.shape[1] * semantics.shape[2]

    present_classes = []

    for class_name in classes_in_questions:
        class_number = label_name_info_inv[class_name]

        class_px_count = torch.sum(semantics == class_number)

        ratio = class_px_count / tot_px_count

        if ratio > min_ratio:
            present_classes.append(class_name)

    return present_classes


def process_yesno_result(result, question_info):
    if "yes" in result or question_info[0] in result:
        processed_result = question_info[0]
    elif "no" in result or question_info[1] in result:
        processed_result = question_info[1]
    else:
        print(
            "Unexpected result to question. Not a yes/no response. Ignoring..."
        )
        processed_result = ""

    return processed_result
