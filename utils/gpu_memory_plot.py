import subprocess
import pynvml 


def get_free_gpu_memory():
    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        ACCEPTABLE_AVAILABLE_MEMORY = 1024
        COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = _output_to_list(
            subprocess.check_output(COMMAND.split()))[1:]
        memory_free_values = [
            int(x.split()[0]) for i, x in enumerate(memory_free_info)
        ]
        return memory_free_values
    except Exception as e:
        print("Could not print GPU memory info: ", e)
        return []


def get_gpu_info():

    UNIT = 1024 * 1024

    pynvml.nvmlInit() 

    gpuDeviceCount = pynvml.nvmlDeviceGetCount() 
    print("GPU num：", gpuDeviceCount)

    for i in range(gpuDeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            i) 

        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle) 

        print("No. %d GPU：" % i, "-" * 30)
        print("free memory：", memoryInfo.free / UNIT, "MB")

    pynvml.nvmlShutdown()  


if __name__ == '__main__':
    print(get_free_gpu_memory())
    print(get_gpu_info())
