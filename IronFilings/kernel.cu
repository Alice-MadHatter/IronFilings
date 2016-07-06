
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust\host_vector.h>
#include <thrust\device_vector.h>

#include <glfw\glfw3.h>
#include <glm\glm.hpp>

using namespace std;
using namespace thrust;
using namespace glm;

class MyData
{
public:
	float scalar;
	vec3 vector;
	__host__ __device__ MyData() : scalar(0.0f), vector(0.0f) {};
};

__global__ void myFunc(MyData* data, int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
		data[index].vector *= data[index].scalar;
}

int main(int argc, char** argv)
{
	/************************************* CUDA Test Begin *************************************/

	// vector size
	const int vecSize = 65536;

	// initialize vectors
	host_vector<MyData> hostVec(vecSize);
	device_vector<MyData> deviceVec(vecSize);
	for (int i = 0; i < vecSize; i++)
	{
		hostVec[i].vector.x = i + 0; hostVec[i].vector.y = i + 1; hostVec[i].vector.z = i + 2;
		hostVec[i].scalar = i;
	}

	// get device property
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int maxThreadNumber = deviceProp.maxThreadsDim[0];
	int maxBlockNumber = deviceProp.maxGridSize[0];

	// memory copy & calculation
	cudaMemcpy(raw_pointer_cast(deviceVec.data()), raw_pointer_cast(hostVec.data()), sizeof(vec3) * hostVec.size(), cudaMemcpyHostToDevice);
	myFunc << <maxBlockNumber, maxThreadNumber >> >(raw_pointer_cast(deviceVec.data()), vecSize);
	cudaMemcpy(raw_pointer_cast(hostVec.data()), raw_pointer_cast(deviceVec.data()), sizeof(vec3) * deviceVec.size(), cudaMemcpyDeviceToHost);

	// print partial result
	cout << "Hello IronFilings !" << endl;
	for (int i = 0; i < 100; i++)
		cout << "(" << hostVec[i].vector.x << ", " << hostVec[i].vector.y << ", " << hostVec[i].vector.z << ") ";
	cout << endl;

	/************************************* CUDA Test End *************************************/

	/************************************* GLFW Test Begin *************************************/

	// initialize GLFW
	if (!glfwInit())
		return -1;

	// create window
	GLFWwindow* window = glfwCreateWindow(256, 256, "Hello IronFilings !", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	// make context current
	glfwMakeContextCurrent(window);

	// main loop
	while (!glfwWindowShouldClose(window)) // until window should close
	{
		// draw triangle
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT);
		glLoadIdentity();
		glTranslatef(0.0f, 0.0f, 0.0f);
		glBegin(GL_TRIANGLES);
		{
			glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
			glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, 0.0f);
			glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(1.0f, -1.0f, 0.0f);
		}
		glEnd();

		// swap buffers
		glfwSwapBuffers(window);

		// poll events
		glfwPollEvents();
	}

	// terminate GLFW
	glfwTerminate();

	/************************************* GLFW Test End *************************************/

	return EXIT_SUCCESS;
}
