# used to clean the record files 

import os

def get_home_directory_with_expanduser():
    return os.path.expanduser("~")

def main():
    # homePath = get_home_directory_with_expanduser()
    homePath = ""
    
    bgemmTime =       "../records/time/abftbgemm.txt"
    bgemmEffecience = "../records/effeciency/abftbgemm.txt"

    gemmTime =       "../records/time/abftgemm.txt"
    gemmEffecience = "../records/effeciency/abftgemm.txt"

    biasTime =       "../records/time/abftBias.txt"
    biasEffecience = "../records/effeciency/abftBias.txt"

    attnTime = "../records/time/attn.txt"

    QTime = "../records/time/Q.txt"
    KTime = "../records/time/K.txt"
    VTime = "../records/time/V.txt"
    ASTime = "../records/time/AS.txt"
    CLTime = "../records/time/CL.txt"
    OUTTime = "../records/time/OUT.txt"

    bgemmCompTime =       "../records/time/abftbgemm_Computing.txt"
    gemmCompTime =       "../records/time/abftgemm_Computing.txt"
    biasCompTime =       "../records/time/abftBias_Computing.txt"

    prepareTime = "../records/time/preparation.txt"

    training_time = "../records/training_time.txt"
    loss = "../records/loss.txt"

    save_time = "../records/save_time.txt"
    load_time = "../records/load_time.txt"

    BGemmC_time = "../records/time/BGemmCorrect.txt"
    BiasC_time = "../records/time/BiasCorrect.txt"

    Cpy_time = "../records/time/cpy.txt"

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
    
    with open((homePath + attnTime), "w") as frTime:
        frTime.truncate(0)

    with open((homePath + QTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + KTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + VTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + ASTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + CLTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + OUTTime), "w") as frTime:
        frTime.truncate(0)

    with open((homePath + bgemmCompTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + gemmCompTime), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + biasCompTime), "w") as frTime:
        frTime.truncate(0)

    with open((homePath + training_time), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + loss), "w") as frTime:
        frTime.truncate(0)

    with open((homePath + prepareTime), "w") as frTime:
        frTime.truncate(0)

    with open((homePath + save_time), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + load_time), "w") as frTime:
        frTime.truncate(0)

    with open((homePath + BGemmC_time), "w") as frTime:
        frTime.truncate(0)
    with open((homePath + BiasC_time), "w") as frTime:
        frTime.truncate(0)

    with open((homePath + Cpy_time), "w") as frTime:
        frTime.truncate(0)

    print("Finish Cleaning Records.")

    return 0

if __name__=="__main__": 
    main() 