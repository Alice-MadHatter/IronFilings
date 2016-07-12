#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <glfw\glfw3.h>
#include <glm\glm.hpp>

using namespace std;
using namespace glm;

typedef unordered_map<int, bool> kmap;

class Application
{
protected:
	static GLFWwindow* window;
	static string title;
	static int width;
	static int height;
	static bool fullscreen;
	static kmap keyboardMap;
	static vec2 mousePosition;

public:
	Application(string title, int width, int height, bool fullscreen);
	virtual ~Application(void);
	void run(void);

protected:
	virtual void initialize(void) {};
	virtual void update(float delta) {};
	virtual void terminate(void) {};
	static void keyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
};
