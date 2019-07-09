

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef WIN32
	#include <unistd.h>
	#include <linux/limits.h>
	void getpath(char* path, int size){
		readlink("/proc/self/exe", path, size);
	}
#else
	#include <Windows.h>
	void getpath(char* path, int size){
		GetModuleFileNameA(nullptr, path, size);
	}
#endif

#define max(a, b)    ((a) > (b) ? (a) : (b))

int main(int argc, char** argv){

	char path[1000];
	getpath(path, sizeof(path));

	char* p = strrchr(path, '.');
	if (p){
		*p = 0;
	}else{
		p = path;
	}
	strcat(path, ".py");

	char cmd[10000] = {0};
	strcpy(cmd, "python");
	p = cmd + strlen(cmd);

	for (int i = 0; i < argc; ++i){
		if (i == 0){
			sprintf(p, " \"%s\"", path);
			p = p + strlen(p);
		}
		else{
			sprintf(p, " \"%s\"", argv[i]);
			p = p + strlen(p);
		}
	}
	//printf("cmd: %s\n", cmd);
	return system(cmd);
}