"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
function classDecorator(params) {
    return function (target) {
        console.log('类装饰器' + params);
    };
}
function propertyDecorator(params) {
    return function (target, attr) {
        console.log('属性装饰器' + params);
    };
}
function methodDecorator(params) {
    return function (target, methodName, desc) {
        console.log('方法装饰器' + params);
    };
}
function paramDecorator(params) {
    return function (target, methodName, paramsIndex) {
        console.log('方法参数装饰器', paramsIndex);
    };
}
let HttpClient = class HttpClient {
    setData(val) {
        console.log(val);
    }
    getData(p1, p2) {
        console.log(p1, p2);
    }
};
__decorate([
    propertyDecorator(1)
], HttpClient.prototype, "url", void 0);
__decorate([
    methodDecorator(2)
], HttpClient.prototype, "setData", null);
__decorate([
    methodDecorator(1),
    methodDecorator(11),
    __param(0, paramDecorator(1)), __param(1, paramDecorator(2))
], HttpClient.prototype, "getData", null);
__decorate([
    propertyDecorator(2)
], HttpClient.prototype, "uuid", void 0);
HttpClient = __decorate([
    classDecorator(1),
    classDecorator(2)
], HttpClient);
let h = new HttpClient();
