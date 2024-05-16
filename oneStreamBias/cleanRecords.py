# used to clean the record files 

import os

def get_home_directory_with_expanduser():
    return os.path.expanduser("~")

def main():
    homePath = get_home_directory_with_expanduser()
    
    bgemmTime =       "/abftbgemm/records/time/abftbgemm.txt"
    bgemmEffecience = "/abftbgemm/records/effeciency/abftbgemm.txt"

    gemmTime =       "/abftbgemm/records/time/abftgemm.txt"
    gemmEffecience = "/abftbgemm/records/effeciency/abftgemm.txt"

    biasTime =       "/abftbgemm/records/time/abftBias.txt"
    biasEffecience = "/abftbgemm/records/effeciency/abftBias.txt"

    with open((homePath + bgemmTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + bgemmEffecience), "w") as frEffecience:
        frEffecience.truncate(0)

    with open((homePath + gemmTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + gemmEffecience), "w") as frEffecience:
        frEffecience.truncate(0)

    with open((homePath + biasTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + biasEffecience), "w") as frEffecience:
        frEffecience.truncate(0)

    print("Finish Cleaning Records.")

    return 0

if __name__=="__main__": 
    main() 