
#include "application.cuh"

GLFWwindow* Application::window = NULL;
string Application::title = "Hello, Iron Filings!";
int Application::width = 1024;
int Application::height = 768;
bool Application::fullscreen = false;
kmap Application::keyboardMap;
vec2 Application::mousePosition(0.0f);

Application::Application(string title, int width, int height, bool fullscreen)
{
	// check if window already exists
	if (window) return;

	// set attributes
	this->title = title;
	this->width = width;
	this->height = height;
	this->fullscreen = fullscreen;

	// initialize glfw
	if (!glfwInit())
	{
		// error check
		cout << "[ERROR] Cannot initialize GLFW" << endl;
		exit(EXIT_FAILURE);
	}

	// get monitor
	GLFWmonitor* monitor = (fullscreen ? glfwGetPrimaryMonitor() : NULL);

	// create window
	window = glfwCreateWindow(width, height, title.c_str(), monitor, NULL);

	// initialize keyboard map
	for (int i = GLFW_KEY_UNKNOWN; i <= GLFW_KEY_LAST; i++)
		keyboardMap[i] = false;

	// set input callback function
	glfwSetKeyCallback(window, keyboardCallback); glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
	glfwSetCursorPosCallback(window, mouseCallback); glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

	// make context current
	glfwMakeContextCurrent(window);

	// turn off vsync
	glfwSwapInterval(0);
}

Application::~Application(void)
{
	// check if window not exists
	if (!window) return;

	// destroy window
	glfwDestroyWindow(window);

	// terminate glfw
	glfwTerminate();

	// reset window
	window = NULL;
}

void Application::run(void)
{
	// initialize process
	initialize();

	// main loop
	double time = glfwGetTime();
	int sec = 0;
	while (!glfwWindowShouldClose(window))
	{
		// update viewport
		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);

		// calculate delta time
		float delta = (float)(glfwGetTime() - time);
		time = glfwGetTime();

		// show FPS in title bar
		if ((int)time > sec)
		{
			sec = (int)time;
			ostringstream oss;
			oss << title << " (FPS: " << (int)(1.0f / delta) << ")";
			glfwSetWindowTitle(window, oss.str().c_str());
		}

		// update process
		update(delta);

		// swap buffers and poll events
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// terminate process
	terminate();
}

void Application::keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// update keyboard map
	keyboardMap[key] = (action == GLFW_PRESS || action == GLFW_REPEAT);
}

void Application::mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	// update mouse position
	mousePosition.x = (float)xpos;
	mousePosition.y = (float)ypos;
}
