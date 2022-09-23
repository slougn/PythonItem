/*
 * @Author: 560130
 * @Date: 2022-09-23 11:18:23
 * @LastEditTime: 2022-09-23 11:33:20
 * @LastEditors: 560130
 * @Description: 
 * @FilePath: /PythonItem/操作系统/ch2.c
 */

#include<stdio.h>
#include<string.h>
#include<stdlib.h>

void print_b(void* pointer,size_t size){
    unsigned long data = *((unsigned long*)pointer);
    int length = size * 8;
    int counter = 0;
    printf("十进制：%lu\n",data);
    printf("二进制： ");
    while(length-- > 0){
        printf("%lu",(data>>length)&0x1);
        counter ++;
        if(counter % 8 == 0){
            printf(" ");
        }
        if (counter % 4 == 0){
            printf(" ");
        }
    }
}

int mian(){
    int x = 0x87654321;
    print_b(&x,sizeof(x));

    return 0;
}