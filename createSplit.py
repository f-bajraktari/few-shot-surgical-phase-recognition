import os
import random
import shutil

def CreateSplit(src_path, dst_path, train_class_list, val_class_list, test_class_list, max_number_videos_test=-1, max_number_videos_train=-1):
    video_count = 0
    count = 1
    train_txt_path = os.path.join(dst_path, "trainlist01.txt")
    val_txt_path = os.path.join(dst_path, "vallist01.txt")
    test_txt_path = os.path.join(dst_path, "testlist01.txt")

    while os.path.exists(train_txt_path) or os.path.exists(val_txt_path) or os.path.exists(test_txt_path):
        count += 1
        train_txt_path = os.path.join(dst_path, f"trainlist{count:02}.txt")
        val_txt_path = os.path.join(dst_path, f"vallist{count:02}.txt")
        test_txt_path = os.path.join(dst_path, f"testlist{count:02}.txt")

    with open(train_txt_path, "w") as file:
        for c in train_class_list:
            c_path = os.path.join(src_path, c)
            videos_per_class_count = 0
            for video in os.listdir(c_path):
                file.write(f"{c}/{video}\n")
                video_count += 1
                videos_per_class_count += 1
                if(max_number_videos_train >= 0 and videos_per_class_count >= max_number_videos_train):
                    break

    with open(val_txt_path, "w") as file:
        for c in val_class_list:
            c_path = os.path.join(src_path, c)
            videos_per_class_count = 0
            for video in os.listdir(c_path):
                file.write(f"{c}/{video}\n")
                video_count += 1
                videos_per_class_count += 1
                if(max_number_videos_test >= 0 and videos_per_class_count >= max_number_videos_test):
                    break

    with open(test_txt_path, "w") as file:
        for c in test_class_list:
            c_path = os.path.join(src_path, c)
            videos_per_class_count = 0
            for video in os.listdir(c_path):
                file.write(f"{c}/{video}\n")
                videos_per_class_count += 1
                if(max_number_videos_test >= 0 and videos_per_class_count >= max_number_videos_test):
                    break
                video_count += 1

    print(f"Train Split created: {train_txt_path}")
    print(f"Validation Split created: {val_txt_path}")
    print(f"Test Split created: {test_txt_path}")
    print(f"Video Count: {video_count}")

def createRandomSplit(src_path, number_Train, number_Val, number_Test):
    class_list = []
    for folder in os.listdir(src_path):
        class_list.append(folder)
    
    if len(class_list) < (number_Train + number_Val + number_Test):
        print("Not enough classes...")
        return
    
    train_class_list = random.sample(class_list, number_Train)
    class_list = [elem for elem in class_list if elem not in train_class_list]
    val_class_list = random.sample(class_list, number_Val)
    class_list = [elem for elem in class_list if elem not in val_class_list]
    test_class_list = random.sample(class_list, number_Test)
    class_list = [elem for elem in class_list if elem not in test_class_list]

def createRandomSplit_fixedTestClass(src_path, number_Train, number_Val, Test_Class_Prefix):
    class_list = []
    for folder in os.listdir(src_path):
        class_list.append(folder)
    
    class_list = [elem for elem in class_list if " " not in elem]
    test_class_list = [elem for elem in class_list if elem.startswith(Test_Class_Prefix + "_")]
    class_list = [elem for elem in class_list if elem not in test_class_list]

    if len(class_list) < (number_Train + number_Val):
        print("Not enough classes...")
        return

    train_class_list = random.sample(class_list, number_Train)
    class_list = [elem for elem in class_list if elem not in train_class_list]
    val_class_list = random.sample(class_list, number_Val)
    class_list = [elem for elem in class_list if elem not in val_class_list]
    
    return train_class_list, val_class_list, test_class_list

def createSplit_fixedSurgeries(src_path, train_surgeries_prefix, val_surgeries_prefix, test_surgeries_prefix):
    class_list = []
    for folder in os.listdir(src_path):
        class_list.append(folder)
    
    class_list = [elem for elem in class_list if " " not in elem]

    test_class_list = [elem for elem in class_list if any(elem.startswith(i + "_") for i in test_surgeries_prefix)]
    class_list = [elem for elem in class_list if elem not in test_class_list]

    train_class_list = [elem for elem in class_list if any(elem.startswith(i + "_") for i in train_surgeries_prefix)]
    class_list = [elem for elem in class_list if elem not in train_class_list]

    val_class_list = [elem for elem in class_list if any(elem.startswith(i + "_") for i in val_surgeries_prefix)]
    class_list = [elem for elem in class_list if elem not in val_class_list]
    
    return train_class_list, val_class_list, test_class_list

def createDataset(train_txt_path, val_txt_path, test_txt_path, src_path, dst_path):
    def copy_folders(txt_path):
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            relative_path = line.strip()
            src_folder = os.path.join(src_path, relative_path)
            dst_folder = os.path.join(dst_path, relative_path)
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dst_folder), exist_ok=True)
            
            # Copy the folder
            if os.path.exists(src_folder):
                shutil.copytree(src_folder, dst_folder)
                print(f"Copy {src_folder} to {dst_folder}")
            else:
                print(f"Source folder {src_folder} does not exist")

    # Copy folders for each split
    copy_folders(train_txt_path)
    copy_folders(val_txt_path)
    copy_folders(test_txt_path)

if __name__ == "__main__":
    src_directory = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets/data/SSV2"
    dst_directory = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets/splits/somethingsomethingv2TrainTestlist"
    train_list, val_list, test_list = createSplit_fixedSurgeries(src_directory, ["S2"], [""], ["C80"])
    CreateSplit(src_directory, dst_directory, train_list, val_list, test_list)
    #src = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets/data/surgicalphasev1_Xx256"
    #dst = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets/data/surgicalphasev2_Xx256"
    #train = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets/splits/surgicalphasev2TrainTestlist/trainlist01.txt"
    #test = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets/splits/surgicalphasev2TrainTestlist/testlist01.txt"
    #val = "/media/robert/Volume/Forschungsarbeit_Robert_Asmussen/05_Data/TRX/video_datasets/splits/surgicalphasev2TrainTestlist/vallist01.txt"
    #createDataset(train, val, test, src, dst)