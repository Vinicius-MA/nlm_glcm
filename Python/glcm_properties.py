from enum import Enum

class Props(Enum):
    CONTRAST        =   {"order":0, "name":"contrast"}
    DISSIMILARITY   =   {"order":1, "name":"dissimilarity"}
    HOMOGENEITY     =   {"order":2, "name":"homogeneity"}
    ENERGY          =   {"order":3, "name":"energy"}
    CORRELATION     =   {"order":4, "name":"correlation"}
    ASM             =   {"order":5, "name":"ASM"}

    def get_list( props, order=False ):
        if order:
            return [prop.value["order"] for prop in props]
        else:
            return [prop.value["name"] for prop in props]
    
    def all( order=False):
        if order:
            return [prop.value["order"] for prop in Props]
        else:
            return [prop.value["name"] for prop in Props ]
