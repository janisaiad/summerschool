import tensorflow as tf
import pytest as pt

def test_fft_ifft_nd():
    # we set up test dimensions
    batch_size = 3
    n_points = 5
    n_modes = 10
    
    # we create random input data
    real_data = tf.random.uniform((batch_size, n_points, n_modes), dtype=tf.float32)  # we use float32 for gpu compatibility
    
    # we convert to complex for fftnd (which requires complex input)
    data = tf.cast(real_data, tf.complex64)  # we convert real data to complex
    
    # we check if gpu is available
    gpus = tf.config.list_physical_devices('GPU')  # we use modern gpu detection
    if len(gpus) > 0:
        with tf.device('/GPU:0'):
            # we perform forward fft (specify fft_length for the axis we're transforming)
            fft = tf.signal.fftnd(data, axes=[-1], fft_length=[n_modes])  # we transform along last axis with specified length
            
            # we perform inverse fft
            ifft = tf.signal.ifftnd(fft, axes=[-1], fft_length=[n_modes])
            
            # we verify reconstruction (compare real parts since we started with real data)
            error = tf.reduce_mean(tf.abs(tf.math.real(data) - tf.math.real(ifft)))
            assert error < 1e-5, f"FFT/IFFT reconstruction error too large: {error}"  # we check reconstruction error
            
            print("FFT/IFFT test passed on GPU")
    else:
        print("No GPU available, skipping GPU test")
        
    # we test real fft (this is better for real input data)
    real_input = tf.abs(real_data)  # we ensure real input
    
    # we perform real fft
    rfft = tf.signal.rfftnd(real_input, axes=[-1], fft_length=[n_modes])
    
    # we perform inverse real fft
    irfft = tf.signal.irfftnd(rfft, axes=[-1], fft_length=[n_modes])
    
    # we verify reconstruction for real case
    real_error = tf.reduce_mean(tf.abs(real_input - irfft))
    assert real_error < 1e-5, f"RFFT/IRFFT reconstruction error too large: {real_error}"  # we check real reconstruction error
    
    print("RFFT/IRFFT test passed")

def test_4d_fft_gpu():
    """we test 4d fft transformation on gpu for scientific computing applications"""
    print("\n=== Testing 4D FFT on GPU ===")
    
    # we set up 4d dimensions (typical for scientific simulations)
    batch_size = 2      # we use batch dimension
    nx, ny, nz = 16, 16, 16  # we create 3d spatial grid
    
    print(f"4D data shape: ({batch_size}, {nx}, {ny}, {nz})")
    
    # we create 4d test data (simulating volumetric data)
    real_data = tf.random.uniform((batch_size, nx, ny, nz), dtype=tf.float32)  # we generate random 4d data
    data_complex = tf.cast(real_data, tf.complex64)  # we convert to complex for fft
    
    # we check gpu availability
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        print("Running 4D FFT on GPU...")
        
        with tf.device('/GPU:0'):
            # we measure gpu memory before
            print(f"GPU Memory: {tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e6:.1f} MB")
            
            # we perform 4d fft on all spatial dimensions (axes 1,2,3)
            print("Forward 4D FFT...")
            fft_4d = tf.signal.fftnd(
                data_complex, 
                axes=[1, 2, 3],  # we transform spatial dimensions
                fft_length=[nx, ny, nz]
            )
            
            # we perform inverse 4d fft
            print("Inverse 4D FFT...")
            ifft_4d = tf.signal.ifftnd(
                fft_4d,
                axes=[1, 2, 3],  # we transform back spatial dimensions
                fft_length=[nx, ny, nz]
            )
            
            # we verify reconstruction
            error_4d = tf.reduce_mean(tf.abs(tf.math.real(data_complex) - tf.math.real(ifft_4d)))
            print(f"4D FFT reconstruction error: {error_4d:.2e}")
            
            assert error_4d < 1e-5, f"4D FFT reconstruction error too large: {error_4d}"
            
            # we test partial 4d fft (only 2 axes)
            print("Partial 4D FFT (2 axes)...")
            fft_partial = tf.signal.fftnd(
                data_complex,
                axes=[2, 3],  # we transform only last 2 dimensions
                fft_length=[ny, nz]
            )
            
            ifft_partial = tf.signal.ifftnd(
                fft_partial,
                axes=[2, 3],
                fft_length=[ny, nz]
            )
            
            error_partial = tf.reduce_mean(tf.abs(tf.math.real(data_complex) - tf.math.real(ifft_partial)))
            print(f"Partial FFT reconstruction error: {error_partial:.2e}")
            
            # we measure final gpu memory
            print(f"GPU Memory after: {tf.config.experimental.get_memory_info('GPU:0')['current'] / 1e6:.1f} MB")
            
            print("✅ 4D FFT test passed on GPU!")
            
    else:
        print("❌ No GPU available for 4D FFT test")
        
    # we also test real 4d fft (more efficient for real data)
    print("\nTesting 4D Real FFT...")
    real_input_4d = tf.abs(real_data)  # we ensure real input
    
    with tf.device('/GPU:0' if len(gpus) > 0 else '/CPU:0'):
        # we perform real 4d fft
        rfft_4d = tf.signal.rfftnd(
            real_input_4d,
            axes=[1, 2, 3],  # we transform all spatial axes
            fft_length=[nx, ny, nz]
        )
        
        # we perform inverse real 4d fft
        irfft_4d = tf.signal.irfftnd(
            rfft_4d,
            axes=[1, 2, 3],
            fft_length=[nx, ny, nz]
        )
        
        # we verify real fft reconstruction
        real_error_4d = tf.reduce_mean(tf.abs(real_input_4d - irfft_4d))
        print(f"4D Real FFT reconstruction error: {real_error_4d:.2e}")
        
        assert real_error_4d < 1e-5, f"4D Real FFT reconstruction error too large: {real_error_4d}"
        
        print("✅ 4D Real FFT test passed!")

if __name__ == "__main__":
    test_fft_ifft_nd()
    test_4d_fft_gpu()
