#pragma once
#include "Public.h"
#include "DatabaseHelper.h"
#include <mysql/mysql.h>

class CMysqlClient
        :public CDatabaseClient
{
public:
    CMysqlClient(const CMysqlClient&) = delete;
    CMysqlClient& operator=(const CMysqlClient&) = delete;
public:
    CMysqlClient() {
        bzero(&m_db, sizeof(m_db));
        m_bInit = false;
    }
    virtual ~CMysqlClient() {
        Close();
    }
public:
    //连接
    virtual int Connect(const KeyValue& args);
    //执行
    virtual int Exec(const Buffer& sql);
    //带结果的执行
    virtual int Exec(const Buffer& sql, Result& result, const _Table_& table);
    //开启事务
    virtual int StartTransaction();
    //提交事务
    virtual int CommitTransaction();
    //回滚事务
    virtual int RollbackTransaction();
    //关闭连接
    virtual int Close();
    //是否连接 true表示连接中 false表示未连接
    virtual bool IsConnected();
private:
    MYSQL m_db;
    bool m_bInit;//默认是false 表示没有初始化 初始化之后，则为true，表示已经连接
private:
    class ExecParam {
    public:
        ExecParam(CMysqlClient* obj, Result& result, const _Table_& table)
                :obj(obj), result(result), table(table)
        {}
        CMysqlClient* obj;
        Result& result;
        const _Table_& table;
    };
};

class _mysql_table_ :
        public _Table_
{
public:
    _mysql_table_() :_Table_() {}
    _mysql_table_(const _mysql_table_& table);
    virtual ~_mysql_table_();
    //返回创建的SQL语句
    virtual Buffer Create();
    //删除表
    virtual Buffer Drop();
    //增删改查
    //TODO:参数进行优化
    virtual Buffer Insert(const _Table_& values);
    virtual Buffer Delete(const _Table_& values);
    //TODO:参数进行优化
    virtual Buffer Modify(const _Table_& values);
    virtual Buffer Query();
    //创建一个基于表的对象
    virtual PTable Copy()const;
    virtual void ClearFieldUsed();
public:
    //获取表的全名
    virtual operator const Buffer() const;
};

class _mysql_field_ :
        public _Field_
{
public:
    _mysql_field_();
    _mysql_field_(
            int ntype,
            const Buffer& name,
            unsigned attr,
            const Buffer& type,
            const Buffer& size,
            const Buffer& default_,
            const Buffer& check
    );
    _mysql_field_(const _mysql_field_& field);
    virtual ~_mysql_field_();
    virtual Buffer Create();
    virtual void LoadFromStr(const Buffer& str);
    //where 语句使用的
    virtual Buffer toEqualExp() const;
    virtual Buffer toSqlStr() const;
    //列的全名
    virtual operator const Buffer() const;
private:
    Buffer Str2Hex(const Buffer& data) const;
    union {
        bool Bool;
        int Integer;
        double Double;
        Buffer* String;
    }Value;
    int nType;
};

#define DECLARE_TABLE_CLASS(name, base) class name:public base { \
public: \
virtual PTable Copy() const {return PTable(new name(*this));} \
name():base(){Name=#name;

#define DECLARE_MYSQL_FIELD(ntype,name,attr,type,size,default_,check) \
{PField field(new _mysql_field_(ntype, #name, attr, type, size, default_, check));FieldDefine.push_back(field);Fields[#name] = field; }

#define DECLARE_TABLE_CLASS_EDN() }};
