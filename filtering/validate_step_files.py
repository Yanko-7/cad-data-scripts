from OCC.Extend.DataExchange import read_step_file
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopExp import topexp
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def is_watertight(shape):
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    for i in range(1, edge_face_map.Size() + 1):
        if edge_face_map.FindFromIndex(i).Size() < 2:
            return False
    return True

def is_ok(step_file):
    try:
        shape = read_step_file(step_file)
        if not BRepCheck_Analyzer(shape).IsValid():
            return False
        if not is_watertight(shape):
            return False

    except Exception as e:
        print(f"Error reading STEP file {step_file}: {e}")
        return False
    return True

def gather_step_files(directory):
    import os
    step_files = []
    for file in os.listdir(directory):
        if file.endswith(".step") or file.endswith(".stp"):
            step_files.append(os.path.join(directory, file))
    return step_files

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_step.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    step_files = gather_step_files(directory)
    total = len(step_files)
    success = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(is_ok, step_file) for step_file in step_files]
        for future in tqdm(as_completed(futures), total=total, desc="Processing STEP files"):
            if future.result():
                success += 1
    print(f"Successfully processed {success}/{total} STEP files")

