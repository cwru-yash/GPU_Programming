Instead of copying code from and to the Host memory ad the device memor, we've something called the Unified Memory or the Virtual memroy which takes care of managing both the memories independently and we as programmer can take care of writing our kernel coe better especially when writing very complex code. 

In our previous code of vecotr addition we had to declare 2 sets of memory pointers:
    Declare pointers to Host(CPU) memory --> We used `malloc` for that
    Declare pointers to Device(GPU) memory --> We used `cudaMallocManaged` for that

But in case of Unified Memory (UM):
    We don't use the above process and isntead we use only a single set of pointers say *a, *b, *c and use `cudaMallocmanaged` we're not going to manage where this data lies at any given time, if it's in the CPU and should be on the GPU then that transfer will automatically happen either that will happen behind the scenes or we will give some hints within our code to say that you should come on the CPu now or you should come on the GPU now.

    These unified memroy pointers -->  int *a, *b, *c, can be accessed either from the CPU side or from the GPU side. 


    we use a new keyword within this code called `cudaDeviceSynchronize();`  --> It says/takes care of that at this point in execution, after the synchronization point all previous events on the GPU have finished. Because when we do an asynchronus launch of our kernel and return from that function that kernel is not yet gauranteed to be finished yet.

    Incase of doing soemthing like cudamalloc after a device launch like we did in our regular vecAdd example, cudamalloc is a synchronous call and all of our things because we're doing it to the same stream whcih is a default stream because we didn't specified any other stream, so it goes to the default stream  it will automatcally serealize all of those CUDA calls, so in that case our cudaMemCpy was also a synchronization barrier but in this case since we don;t have ny more `memcpy` or `cudamemcpy` anymore using unified memory we've to explicitly esay that I need to make sure that everything is done, and then if we try to access the data on the vectorADDUM kernel then we're trying to concurrently accesing the data that's on the GPu and the CPU at the same time and then we end  up with this race condition of who should be accessing what at what time so for that case of vectorAdd only the GPU should have access to the data while it's doing that additon. 

    When vector add is successfully done and we then again see that the code is similar to our regular vecAdd code by checking the answer that we've got  by running the chec_answer() test function, otherwise it will throw an assert.

    Important Note: Most of our code looks the same with the previous version of Host and device memory allocation explicilly handled by us, just instead of doing it twice for and within both the host and the device we're doing it once by using the unified memory.



Now How do we imporve performance ?

when we started GPU executionwe explicitlty used cudaMemcpy to cpy our data to and from the GPU and we knw that our data will be going to be on the GPU, but in thsi case we did no such thing, so what happens is the GPU starts upo and says oh I don't have any of this data so it does some thing called page faulting -- which is nothing but page in memroy from the CPU to GPU which is just transferring the memory behind the scenes and get it page by page.

The size of this pages will be different, quite sure what the page size is on the GPUs but the CPUs are generally 4k so it'll be 4Kb of granuality transfred over to the GPU.

We can use something called the Prefetching and speed this up by giving hints as to when the data should be where and we do this by prefetching so prefetching just says behind the scenes you can transfer data while we're doing other things so in this case we can call `cudaMemPrefetchAsync(a, bytes, id)` --> this says prefetch a of size bytes to the GPU/CPU with the device id as `id`. what this does is it say asynchronously start transferring data ahead of time 

LAso we get the id from this function calleed cudaGetDevice(id), incase of multiple GPUs there'll be multiple device Id's to sort between but not rigt now.

We can also start prefetching the device memrory to the CPU, prefetching the device back or that of to the device (basically transferring back to the device)., `cudamemPrefetchAsync(c, bytes cudaDeviceId);`.

We've something called the cudaCpuDeviceId  and we don't need to calcualte the device Id it automatically knows it it's a built-in so we'll just say `cudamemPrefetchAsync(c, bytes cudaDeviceId);`, this basically means I need c of size bytes and I need it on the CPU now.