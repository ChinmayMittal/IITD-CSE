from torch import nn



class BaseAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward():
        raise NotImplementedError

    def get_action():
        raise NotImplementedError
    
    def update():
        raise NotImplementedError
    
        

