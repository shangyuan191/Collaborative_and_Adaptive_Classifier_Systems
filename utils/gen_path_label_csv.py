import os
import csv
import pandas as pd
def gen_csv(path):
    folders=os.listdir(path)
    folders=sorted(folders)
    label_num=1
    folderName_label_mapping={}
    data=[]
    for folder in folders:
        if folder not in folderName_label_mapping:
            folderName_label_mapping[folder]=label_num
            label_num+=1
            folder_path=os.path.join(path,folder)
            for img in os.listdir(folder_path):
                img_path=os.path.join(folder_path,img)
                data.append([img_path,folderName_label_mapping[folder]])
    df=pd.DataFrame(data,columns=["path","label"])
    df.to_csv("./path_label.csv",index=False)
        
        
            

if __name__ == "__main__":
    path="../datasets"
    gen_csv(path)