from tasks.factory_task_peg_hole_pick import FactoryTaskPegHolePick
from tasks.factory_task_peg_hole_place import FactoryTaskPegHolePlace
from tasks.factory_task_peg_hole_insert import FactoryTaskPegHoleInsert

# Mappings from strings to environments
isaacgym_task_map = {
    "FactoryTaskPegHolePick": FactoryTaskPegHolePick,
    "FactoryTaskPegHoleInsert": FactoryTaskPegHoleInsert,
    "FactoryTaskPegHolePlace": FactoryTaskPegHolePlace,

}