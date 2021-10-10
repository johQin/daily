package com.sifangtech.boot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 注解告诉springboot这是一个springboot应用
 * */
@SpringBootApplication
public class mainApplication {
    public static void main(String[] args) {
        SpringApplication.run(mainApplication.class,args);
    }
}
