settings = {
    "setting1": {
        "loss": "Loss_v2",
        "architecture": "Yolo_V1",
        "model": "DecayByBatch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting1.h5",
        "loss_file": "losses-setting1.txt",
        "batch_size": 30,
        "epochs": 50
    },
    "setting2": {
        "loss": "Loss_v3",
        "architecture": "Yolo_V1",
        "model": "DecayByBatch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting2.h5",
        "loss_file": "losses-setting2.txt",
        "batch_size": 30,
        "epochs": 50
    },
     "setting3": {
        "loss": "Loss_v2",
        "architecture": "Yolo_V2",
        "model": "DecayByBatch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting3.h5",
        "loss_file": "losses-setting3.txt",
        "batch_size": 30,
        "epochs": 50
    },
     "setting4": {
        "loss": "Loss_v3",
        "architecture": "Yolo_V2",
        "model": "DecayByBatch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting4.h5",
        "loss_file": "losses-setting4.txt",
        "batch_size": 30,
        "epochs": 50
    },
    "setting5": {
        "loss": "Loss_v2",
        "architecture": "Yolo_V1",
        "model": "DecayByEpoch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting5.h5",
        "loss_file": "losses-setting5.txt",
        "batch_size": 30,
        "epochs": 50
    },
    "setting6": {
        "loss": "Loss_v3",
        "architecture": "Yolo_V1",
        "model": "DecayByEpoch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting6.h5",
        "loss_file": "losses-setting6.txt",
        "batch_size": 30,
        "epochs": 50
    },
     "setting7": {
        "loss": "Loss_v2",
        "architecture": "Yolo_V2",
        "model": "DecayByEpoch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting7.h5",
        "loss_file": "losses-setting7.txt",
        "batch_size": 30,
        "epochs": 50
    },
     "setting8": {
        "loss": "Loss_v3",
        "architecture": "Yolo_V2",
        "model": "DecayByEpoch", 
        "weight_Classification_loss": 100,
        "weight_Object_loss": 1000,
        "weight_Localization_loss": 1,
        "lr": 0.01,
        "decay": 0.01,
        "weight_file": "weight-setting8.h5",
        "loss_file": "losses-setting8.txt",
        "batch_size": 30,
        "epochs": 50
    }
}

setting = settings["setting8"]