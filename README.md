[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/0ZRwC3uv)
# Paralelni Algoritmi, treci projekat, 2024

## Studenti
Nemanja Vujic

## Urađeno
U prvom delu global_to_shared su uradjene stavke 1 i 2
U drugom fajlu simulated_annealing.py je pokusaj simulated_annealing algoritma za stavke 3 i 4

## AI
# mem alloc
    nbt = image_flat.nbytes
    print(nbt)
    image_gpu = cuda.mem_alloc(nbt)
    cuda.memcpy_htod(image_gpu, image_flat)
    
    total_energy_gpu = cuda.mem_alloc(np.dtype(np.int32).itemsize)  
    cuda.memset_d32(total_energy_gpu, 0, 1) 
Is there a different way to allocate memory


LogicError: cuModuleLoadDataEx failed: an illegal memory access was encountered - 

Do you know what are some ways of fixing this error:
Traceback (most recent call last):
  File "C:\Users\necav\PycharmProjects\paralelni-algoritmi-treci-projekat-tim1_nemanjavujic\global_to_shared.py", line 58, in <module>
    energy = calculate_image_energy(image)
  File "C:\Users\necav\PycharmProjects\paralelni-algoritmi-treci-projekat-tim1_nemanjavujic\global_to_shared.py", line 27, in calculate_image_energy
    mod = SourceModule("""
  File "C:\Users\necav\anaconda3\envs\paralelni-algoritmi-treci-projekat-tim1_nemanjavujic\lib\site-packages\pycuda\compiler.py", line 355, in __init__
    cubin = compile(
  File "C:\Users\necav\anaconda3\envs\paralelni-algoritmi-treci-projekat-tim1_nemanjavujic\lib\site-packages\pycuda\compiler.py", line 304, in compile
    return compile_plain(source, options, keep, nvcc, cache_dir, target)
  File "C:\Users\necav\anaconda3\envs\paralelni-algoritmi-treci-projekat-tim1_nemanjavujic\lib\site-packages\pycuda\compiler.py", line 154, in compile_plain
    raise CompileError(
pycuda.driver.CompileError: nvcc compilation of C:\Users\necav\AppData\Local\Temp\tmpsieeb2pb\kernel.cu failed
[command: nvcc --cubin -arch sm_61 -m64 -IC:\Users\necav\anaconda3\envs\paralelni-algoritmi-treci-projekat-tim1_nemanjavujic\lib\site-packages\pycuda\cuda kernel.cu]

What are some ways that i can check cuda code errors

probability of random acceptance without curand in cuda code

LogicError: cuMemcpyHtoD failed: misaligned address when sending an array of swaps, what are some fixes
