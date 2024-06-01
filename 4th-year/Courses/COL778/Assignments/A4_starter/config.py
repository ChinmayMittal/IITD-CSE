
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DONT CHANGE THE OVERALL STRUCTURE OF THE DICTIONARY. 

configs = {
    

    'Hopper-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': None,
                'n_layers': None,
                'batch_size': None, 
            },
            "num_iteration": 100,

        },

        "RL":{
            #You can add or change the keys here
               "hyperparameters": {
                'hidden_size': None,
                'n_layers': None,
                'batch_size': None, 
                
            },
            "num_iteration": 100,
        },
    },
    
    
    'Ant-v4': {
        "imitation":{
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': None,
                'n_layers': None,
                'batch_size': None, 
            },
            "num_iteration": 100,

        },
        
        "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": None,
                },            
        },
    },
    
    'PandaPush-v3': {
        "SB3":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": None,
                },            
        },
    },

}