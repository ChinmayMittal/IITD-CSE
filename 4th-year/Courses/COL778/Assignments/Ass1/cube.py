import numpy as np
class faces:
    'main class of the cube'
class edges:
    'class for defining edges'
class corners:
    'class for defining corners'
class movements:
    """
    defining the rotation matrices
    """
   
    def __init__(self,location):
        self.coords = np.array(location)
        FiB = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [ 0, 0, 1]])
        FBi= np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])

        # 90 degree rotations in the XZ plane (around the y-axis when viewed pointing toward you).
        UiD = np.array([[0, 0, -1],
                        [0, 1, 0],
                        [1, 0, 0]])
        UDi = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [-1, 0, 0]])

        # 90 degree rotations in the YZ plane (around the x-axis when viewed pointing toward you).
        RLi = np.array([[1, 0, 0],
                        [0, 0, -1],
                        [0, 1, 0]])
        RiL = np.array([[1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]])
        self.moves = {
            "FBI":FBi,"FIB":FiB,"RLI":RLi,"RIL":RiL,"UDI":UDi,"UiD":UiD
        }
    def _move(self,move):
         # 90 degree rotations in the XY plane. CW is clockwise, CC is counter-clockwise.
        self.coords = np.dot(self.coords,self.moves[move])
if __name__ == '__main__':
    WGR = movements([1,1,1])
    WGR._move("FIB")
    print(WGR.coords)
    WG = movements([0,1,1])
    WG._move("UDI")
    print(WG.coords)
    WG._move("RIL")
    print(WG.coords)
