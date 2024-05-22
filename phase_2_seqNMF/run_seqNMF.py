import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define input parameters
l = 100  # length of timelag
k = 5  # number of factors
lambda_ortho = 0.0001
lambda_bin = 0.01
lambda_1 = 0.01

# Run the MATLAB script
eng.NMF_kitchen(l, k, lambda_1, lambda_ortho, lambda_bin, nargout=0)