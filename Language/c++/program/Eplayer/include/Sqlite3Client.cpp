//
// Created by buntu on 2023/8/15.
//
#include "Sqlite3Client.h"
#include "Logger.h"

// 数据库的操作


int CSqlite3Client::Connect(const KeyValue& args)
{
    auto it = args.find("host");
    if (it == args.end())return -1;     // 看是否有host字段，sqlite3非常小,轻量级,就几百K大小;不需要用户名,密码,直接就可以对数据库进行操作。
    if (m_db != NULL)return -2;         // 数据库对象，一旦连上了就不能再连了
    int ret = sqlite3_open(it->second, &m_db);
    if (ret != 0) {
        TRACEE("connect failed:%d [%s]", ret, sqlite3_errmsg(m_db));
        return -3;
    }
    return 0;
}
//不带结果的执行
int CSqlite3Client::Exec(const Buffer& sql)
{
    printf("sql={%s}\n", (char*)sql);
    if (m_db == NULL)return -1;     // 没有连接成功
    int ret = sqlite3_exec(m_db, sql, NULL, this, NULL);
    if (ret != SQLITE_OK) {
        printf("sql={%s}\n", (char*)sql);
        printf("Exec failed:%d [%s]\n", ret, sqlite3_errmsg(m_db));
        return -2;
    }
    return 0;
}
// 带结果的执行
int CSqlite3Client::Exec(const Buffer& sql, Result& result, const _Table_& table)
{
    char* errmsg = NULL;
    if (m_db == NULL)return -1;
    printf("sql={%s}\n", (char*)sql);
    ExecParam param(this, result, table);
    int ret = sqlite3_exec(m_db, sql,
                           &CSqlite3Client::ExecCallback, (void*)&param, &errmsg);      // param为callback的第一个参数,
                           // 静态函数CSqlite3Client::ExecCallback 传的是 ExecParam param，param包含有this，
                           // 在这个静态函数ExecCallback里会通过param的this，调用非静态的成员函数ExecCallback
    if (ret != SQLITE_OK) {
        printf("sql={%s}\n", (char *)sql);
        printf("Exec failed:%d [%s]\n", ret, errmsg);
        if (errmsg)sqlite3_free(errmsg);
        return -2;
    }
    if (errmsg)sqlite3_free(errmsg);
    return 0;
}
// 开始事务
int CSqlite3Client::StartTransaction()
{
    if (m_db == NULL)return -1;
    int ret = sqlite3_exec(m_db, "BEGIN TRANSACTION", 0, 0, NULL);
    if (ret != SQLITE_OK) {
        TRACEE("sql={BEGIN TRANSACTION}");
        TRACEE("BEGIN failed:%d [%s]", ret, sqlite3_errmsg(m_db));
        return -2;
    }
    return 0;
}
//提交事务
int CSqlite3Client::CommitTransaction()
{
    if (m_db == NULL)return -1;
    int ret = sqlite3_exec(m_db, "COMMIT TRANSACTION", 0, 0, NULL);
    if (ret != SQLITE_OK) {
        TRACEE("sql={COMMIT TRANSACTION}");
        TRACEE("COMMIT failed:%d [%s]", ret, sqlite3_errmsg(m_db));
        return -2;
    }
    return 0;
}
// 事务回滚
int CSqlite3Client::RollbackTransaction()
{
    if (m_db == NULL)return -1;
    int ret = sqlite3_exec(m_db, "ROLLBACK TRANSACTION", 0, 0, NULL);
    if (ret != SQLITE_OK) {
        TRACEE("sql={ROLLBACK TRANSACTION}");
        TRACEE("ROLLBACK failed:%d [%s]", ret, sqlite3_errmsg(m_db));
        return -2;
    }
    return 0;
}
// 关闭数据库连接
int CSqlite3Client::Close()
{
    if (m_db == NULL)return -1;
    int ret = sqlite3_close(m_db);
    if (ret != SQLITE_OK) {
        TRACEE("Close failed:%d [%s]", ret, sqlite3_errmsg(m_db));
        return -2;
    }
    m_db = NULL;
    return 0;
}
// 返回数据库连接状态，true连接，false未连接
bool CSqlite3Client::IsConnected()
{
    return m_db != NULL;
}

// 静态的回调函数会通过arg，去调用非静态的回调函数
int CSqlite3Client::ExecCallback(void* arg, int count, char** values, char** names)
{
    ExecParam* param = (ExecParam*)arg;
    // 调用非静态的回调函数
    return param->obj->ExecCallback(param->result, param->table, count, names, values);
}

int CSqlite3Client::ExecCallback(Result& result, const _Table_& table, int count, char** names, char** values)
// count存放有多少个字段，names存放字段名的列表，values存放每个字段的值
{
    //
    PTable pTable = table.Copy();
    if (pTable == nullptr) {
        printf("table %s error!\n", (const char*)(Buffer)table);
        return -1;
    }

    // 将每个字段的数据
    for (int i = 0; i < count; i++) {
        Buffer name = names[i];
        auto it = pTable->Fields.find(name);
        if (it == pTable->Fields.end()) {
            printf("table %s error!\n", (const char*)(Buffer)table);
            return -2;
        }
        if (values[i] != NULL)
            it->second->LoadFromStr(values[i]);     //将字符串转为对应的类型
    }
    result.push_back(pTable);
    return 0;
}



// 表的操作

// 表对象拷贝构造
_sqlite3_table_::_sqlite3_table_(const _sqlite3_table_& table)
{
    Database = table.Database;
    Name = table.Name;
    for (size_t i = 0; i < table.FieldDefine.size(); i++)
    {
        PField field = PField(new _sqlite3_field_(*
                                                          (_sqlite3_field_*) table.FieldDefine[i].get()
                                                     )
                             );
        FieldDefine.push_back(field);
        Fields[field->Name] = field;
    }
}

_sqlite3_table_::~_sqlite3_table_()
{}

// 返回创建表的语句
Buffer _sqlite3_table_::Create()
{	//CREATE TABLE IF NOT EXISTS 表全名 (列定义,……);
    //表全名 = 数据库.表名
    Buffer sql = "CREATE TABLE IF NOT EXISTS " + (Buffer)*this + "(\r\n";
    for (size_t i = 0; i < FieldDefine.size(); i++) {
        if (i > 0)sql += ",";
        sql += FieldDefine[i]->Create();
    }
    sql += ");";
    TRACEI("sql = %s", (char*)sql);
    return sql;
}

// 返回删除表的语句
Buffer _sqlite3_table_::Drop()
{
    Buffer sql = "DROP TABLE " + (Buffer)*this + ";";
    TRACEI("sql = %s", (char*)sql);
    return sql;
}

// 返回插入记录的语句
Buffer _sqlite3_table_::Insert(const _Table_& values)
{	//INSERT INTO 表全名 (列1,...,列n)
    //VALUES(值1,...,值n);
    Buffer sql = "INSERT INTO " + (Buffer)*this + " (";
    bool isfirst = true;
    for (size_t i = 0; i < values.FieldDefine.size(); i++) {
        if (values.FieldDefine[i]->Condition & SQL_INSERT) {
            if (!isfirst)sql += ",";
            else isfirst = false;
            sql += (Buffer)*values.FieldDefine[i];
        }
    }
    sql += ") VALUES (";
    isfirst = true;
    for (size_t i = 0; i < values.FieldDefine.size(); i++) {
        if (values.FieldDefine[i]->Condition & SQL_INSERT) {
            if (!isfirst)sql += ",";
            else isfirst = false;
            sql += values.FieldDefine[i]->toSqlStr();
        }
    }
    sql += " );";
    TRACEI("sql = %s", (char*)sql);
    return sql;
}

Buffer _sqlite3_table_::Delete(const _Table_& values)
{// DELETE FROM 表全名 WHERE 条件
    Buffer sql = "DELETE FROM " + (Buffer)*this + " ";
    Buffer Where = "";
    bool isfirst = true;
    for (size_t i = 0; i < FieldDefine.size(); i++) {
        if (FieldDefine[i]->Condition & SQL_CONDITION) {
            if (!isfirst)Where += " AND ";
            else isfirst = false;
            Where += (Buffer)*FieldDefine[i] + "=" + FieldDefine[i]->toSqlStr();
        }
    }
    if (Where.size() > 0)
        sql += " WHERE " + Where;
    sql += ";";
    TRACEI("sql = %s", (char*)sql);
    return sql;
}

Buffer _sqlite3_table_::Modify(const _Table_& values)
{
    //UPDATE 表全名 SET 列1=值1 , ... , 列n=值n [WHERE 条件];
    Buffer sql = "UPDATE " + (Buffer)*this + " SET ";
    bool isfirst = true;
    for (size_t i = 0; i < values.FieldDefine.size(); i++) {
        if (values.FieldDefine[i]->Condition & SQL_MODIFY) {
            if (!isfirst)sql += ",";
            else isfirst = false;
            sql += (Buffer)*values.FieldDefine[i] + "=" + values.FieldDefine[i]->toSqlStr();
        }
    }

    Buffer Where = "";
    for (size_t i = 0; i < values.FieldDefine.size(); i++) {
        if (values.FieldDefine[i]->Condition & SQL_CONDITION) {
            if (!isfirst)Where += " AND ";
            else isfirst = false;
            Where += (Buffer)*values.FieldDefine[i] + "=" + values.FieldDefine[i]->toSqlStr();
        }
    }
    if (Where.size() > 0)
        sql += " WHERE " + Where;
    sql += " ;";
    TRACEI("sql = %s", (char*)sql);
    return sql;
}

Buffer _sqlite3_table_::Query(const Buffer& condition)
// 这里作了一个最简单的全查。Query本来是可以带参数，还可以按条件查询等
{//SELECT 列名1 ,列名2 ,... ,列名n FROM 表全名;
    Buffer sql = "SELECT ";
    for (size_t i = 0; i < FieldDefine.size(); i++)
    {
        if (i > 0)sql += ',';
        sql += '"' + FieldDefine[i]->Name + "\" ";
    }
    sql += " FROM " + (Buffer)*this + " ";
    if (condition.size() > 0) {
        sql += " WHERE " + condition;
    }
    sql += ";";
    printf("sql = %s\n", (char*)sql);
    return sql;
}

PTable _sqlite3_table_::Copy() const
{
    return PTable(new _sqlite3_table_(*this));
}

void _sqlite3_table_::ClearFieldUsed()
{
    for (size_t i = 0; i < FieldDefine.size(); i++) {
        FieldDefine[i]->Condition = 0;
    }
}
// 返回数据表名，如果有数据库，顺带返回数据库名
_sqlite3_table_::operator const Buffer() const
{
    Buffer Head;
    if (Database.size())
        Head = '"' + Database + "\".";
    return Head + '"' + Name + '"';
}


//字段的操作


_sqlite3_field_::_sqlite3_field_()
        :_Field_() {
    nType = TYPE_NULL;
    Value.Double = 0.0;
}

_sqlite3_field_::_sqlite3_field_(int ntype, const Buffer& name, unsigned attr, const Buffer& type, const Buffer& size, const Buffer& default_, const Buffer& check)
{
    nType = ntype;
    switch (ntype)
    {
        case TYPE_VARCHAR:
        case TYPE_TEXT:
        case TYPE_BLOB:
            Value.String = new Buffer();
            break;
    }

    Name = name;
    Attr = attr;
    Type = type;
    Size = size;
    Default = default_;
    Check = check;
}

_sqlite3_field_::_sqlite3_field_(const _sqlite3_field_& field)
{
    nType = field.nType;
    switch (field.nType)
    {
        case TYPE_VARCHAR:
        case TYPE_TEXT:
        case TYPE_BLOB:
            Value.String = new Buffer();
            *Value.String = *field.Value.String;
            break;
    }

    Name = field.Name;
    Attr = field.Attr;
    Type = field.Type;
    Size = field.Size;
    Default = field.Default;
    Check = field.Check;
}

_sqlite3_field_::~_sqlite3_field_()
{
    switch (nType)
    {
        case TYPE_VARCHAR:
        case TYPE_TEXT:
        case TYPE_BLOB:
            if (Value.String) {
                Buffer* p = Value.String;
                Value.String = NULL;
                delete p;
            }
            break;
    }
}

// 声明字段的约束
Buffer _sqlite3_field_::Create()
{	//"名称" 类型 属性
    Buffer sql = '"' + Name + "\" " + Type + " ";
    if (Attr & NOT_NULL) {
        sql += " NOT NULL ";
    }
    if (Attr & DEFAULT) {
        sql += " DEFAULT " + Default + " ";
    }
    if (Attr & UNIQUE) {
        sql += " UNIQUE ";
    }
    if (Attr & PRIMARY_KEY) {
        sql += " PRIMARY KEY ";
    }
    if (Attr & CHECK) {
        sql += " CHECK( " + Check + ") ";
    }
    if (Attr & AUTOINCREMENT) {
        sql += " AUTOINCREMENT ";
    }
    return sql;
}

// 根据字段的类型，将str转换为对应类型的值
void _sqlite3_field_::LoadFromStr(const Buffer& str)
{
    switch (nType)
    {
        case TYPE_NULL:
            break;
        case TYPE_BOOL:
        case TYPE_INT:
        case TYPE_DATETIME:
            Value.Integer = atoi(str);
            break;
        case TYPE_REAL:
            Value.Double = atof(str);
            break;
        case TYPE_VARCHAR:
        case TYPE_TEXT:
            *Value.String = str;
            break;
        case TYPE_BLOB:
            *Value.String = Str2Hex(str);
            break;
        default:
            TRACEW("type=%d", nType);
            break;
    }
}
// 用在where子句中
Buffer _sqlite3_field_::toEqualExp() const
{
    Buffer sql = (Buffer)*this + " = ";
    std::stringstream ss;
    switch (nType)
    {
        case TYPE_NULL:
            sql += " NULL ";
            break;
        case TYPE_BOOL:
        case TYPE_INT:
        case TYPE_DATETIME:
            ss << Value.Integer;
            sql += ss.str() + " ";
            break;
        case TYPE_REAL:
            ss << Value.Double;
            sql += ss.str() + " ";
            break;
        case TYPE_VARCHAR:
        case TYPE_TEXT:
        case TYPE_BLOB:
            sql += '"' + *Value.String + "\" ";
            break;
        default:
            TRACEW("type=%d", nType);
            break;
    }
    return sql;
}

Buffer _sqlite3_field_::toSqlStr() const
{
    Buffer sql = "";
    std::stringstream ss;
    switch (nType)
    {
        case TYPE_NULL:
            sql += " NULL ";
            break;
        case TYPE_BOOL:
        case TYPE_INT:
        case TYPE_DATETIME:
            ss << Value.Integer;
            sql += ss.str() + " ";
            break;
        case TYPE_REAL:
            ss << Value.Double;
            sql += ss.str() + " ";
            break;
        case TYPE_VARCHAR:
        case TYPE_TEXT:
        case TYPE_BLOB:
            sql += '"' + *Value.String + "\" ";
            break;
        default:
            TRACEW("type=%d", nType);
            break;
    }
    return sql;
}

_sqlite3_field_::operator const Buffer() const
{
    return '"' + Name + '"';
}
// 将字符串转为2进制
Buffer _sqlite3_field_::Str2Hex(const Buffer& data) const
{
    const char* hex = "0123456789ABCDEF";
    std::stringstream ss;
    for (auto ch : data)
        ss << hex[(unsigned char)ch >> 4] << hex[(unsigned char)ch & 0xF];
    return ss.str();
}

// 基本表的使用，但这样写有些复杂，还需要简化
/*
class user_test :public _sqlite3_table_
{
public:
	virtual PTable Copy() const {
		return PTable(new user_test(*this));
	}
	user_test() :_sqlite3_table_() {
		Name = "user_test";
		// 加第一个字段
		{
			PField field(new _sqlite3_field_(TYPE_INT, "user_id", NOT_NULL | PRIMARY_KEY | AUTOINCREMENT, "INT", "", "", ""));
			FieldDefine.push_back(field);
			Fields["user_id"] = field;
		}
		// 加第二个字段
		{
			PField field(new _sqlite3_field_(TYPE_VARCHAR, "user_qq", NOT_NULL | PRIMARY_KEY | AUTOINCREMENT, "VARCHAR", "(15)", "", ""));
			FieldDefine.push_back(field);
			Fields["user_qq"] = field;
		}
	}
};*/


