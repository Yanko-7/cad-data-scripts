from utils import load_and_filter_step, write_step_file, preprocess_shape

step_path = "example.step"
shape = load_and_filter_step(step_path)
shape = preprocess_shape(shape)
write_step_file(shape, "example_processed.stp")