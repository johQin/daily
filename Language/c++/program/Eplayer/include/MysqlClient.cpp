#include "MysqlClient.h"
#include <sstream>

int CMysqlClient::Connect(const KeyValue& args)
{
    if (m_bInit)return -1;
    MYSQL* ret = mysql_init(&m_db);
    if (ret == NULL)return -2;
    ret = mysql_real_connect(&m_db,
                             args.at("host"), args.at("user"),          // map at查找，找到就返回val，找不到就报错
                             args.at("password"), args.at("db"),
                             atoi(args.at("port")),
                             NULL, 0);
    if ((ret == NULL) && (mysql_errno(&m_db) != 0)) {
        printf("%s %s\n", __FUNCTION__, mysql_errno(&m_db));
        mysql_close(&m_db);
        bzero(&m_db, sizeof(m_db));
        return -3;
    }
    m_bInit = true;
    return 0;
}

int CMysqlClient::Exec(const Buffer& sql)
{
    if (!m_bInit)return -1;
    int ret = mysql_real_query(&m_db, sql, sql.size());
    if (ret != 0) {
        printf("%s %s\n", __FUNCTION__, mysql_errno(&m_db));
        return -2;
    }
    return 0;
}

int CMysqlClient::Exec(const Buffer& sql, Result& result, const _Table_& table)
{
    if (!m_bInit)return -1;
    int ret = mysql_real_query(&m_db, sql, sql.size());
    if (ret != 0) {
        printf("%s %s\n", __FUNCTION__, mysql_errno(&m_db));
        return -2;
    }
    MYSQL_RES* res = mysql_store_result(&m_db);
    MYSQL_ROW row;
    unsigned num_fields = mysql_num_fields(res);
    while ((row = mysql_fetch_row(res)) != NULL) {
        PTable pt = table.Copy();
        for (unsigned i = 0; i < num_fields; i++) {
            if (row[i] != NULL) {
                pt->FieldDefine[i]->LoadFromStr(row[i]);
            }
        }
        result.push_back(pt);
    }
    return 0;
}

int CMysqlClient::StartTransaction()
{
    if (!m_bInit)return -1;
    int ret = mysql_real_query(&m_db, "BEGIN", 6);
    if (ret != 0) {
        printf("%s %s\n", __FUNCTION__, mysql_errno(&m_db));
        return -2;
    }
    return 0;
}

int CMysqlClient::CommitTransaction()
{
    if (!m_bInit)return -1;
    int ret = mysql_real_query(&m_db, "COMMIT", 7);
    if (ret != 0) {
        printf("%s %s\n", __FUNCTION__, mysql_errno(&m_db));
        return -2;
    }
    return 0;
}

int CMysqlClient::RollbackTransaction()
{
    if (!m_bInit)return -1;
    int ret = mysql_real_query(&m_db, "ROLLBACK", 9);
    if (ret != 0) {
        printf("%s %s\n", __FUNCTION__, mysql_errno(&m_db));
        return -2;
    }
    return 0;
}

int CMysqlClient::Close()
{
    if (m_bInit) {
        m_bInit = false;
        mysql_close(&m_db);
        bzero(&m_db, sizeof(m_db));
    }
    return 0;
}

bool CMysqlClient::IsConnected()
{
    return m_bInit;
}

_mysql_table_::_mysql_table_(const _mysql_table_& table)
{
    Database = table.Database;
    Name = table.Name;
    for (size_t i = 0; i < table.FieldDefine.size(); i++)
    {
        PField field = PField(new _mysql_field_(*
                                                        (_mysql_field_*)table.FieldDefine[i].get()));
        FieldDefine.push_back(field);
        Fields[field->Name] = field;
    }
}

_mysql_table_::~_mysql_table_()
{}

Buffer _mysql_table_::Create()
{	//CREATE TABLE IF NOT EXISTS 表全名 (列定义,..., PRIMARY KEY `主键列名` ,UNIQUE INDEX `列名_UNIQUE` (列名 ASC) VISIBLE );
    Buffer sql = "CREATE TABLE IF NOT EXISTS " + (Buffer)*this + " (\r\n";
    for (unsigned i = 0; i < FieldDefine.size(); i++)
    {
        if (i > 0)sql += ",\r\n";
        sql += FieldDefine[i]->Create();
        if (FieldDefine[i]->Attr & PRIMARY_KEY) {
            sql += ",\r\n PRIMARY KEY (`" + FieldDefine[i]->Name + "`)";
        }
        if (FieldDefine[i]->Attr & UNIQUE) {
            sql += ",\r\n UNIQUE INDEX `" + FieldDefine[i]->Name + "_UNIQUE` (";
            sql += (Buffer)*FieldDefine[i] + " ASC) VISIBLE ";
        }
    }
    sql += ");";
    return sql;
}

Buffer _mysql_table_::Drop()
{
    return "DROP TABLE" + (Buffer)*this;
}

Buffer _mysql_table_::Insert(const _Table_& values)
{// INSERT INTO 表全名 (列名,...)VALUES(值,...);
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
    printf("sql = %s\n", (char*)sql);
    return sql;
}

Buffer _mysql_table_::Delete(const _Table_& values)
{
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
    printf("sql = %s\r\n", (char*)sql);
    return sql;
}

Buffer _mysql_table_::Modify(const _Table_& values)
{
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
    printf("sql = %s\n", (char*)sql);
    return sql;
}

Buffer _mysql_table_::Query()
{
    Buffer sql = "SELECT ";
    for (size_t i = 0; i < FieldDefine.size(); i++)
    {
        if (i > 0)sql += ',';
        sql += '`' + FieldDefine[i]->Name + "` ";
    }
    sql += " FROM " + (Buffer)*this + ";";
    printf("sql = %s\n", (char*)sql);
    return sql;
}

PTable _mysql_table_::Copy() const
{
    return PTable(new _mysql_table_(*this));
}

void _mysql_table_::ClearFieldUsed()
{
    for (size_t i = 0; i < FieldDefine.size(); i++) {
        FieldDefine[i]->Condition = 0;
    }
}

_mysql_table_::operator const Buffer() const
{
    Buffer Head;
    if (Database.size())
        Head = '`' + Database + "`.";
    return Head + '`' + Name + '`';
}

_mysql_field_::_mysql_field_() :_Field_()
{
    nType = TYPE_NULL;
    Value.Double = 0.0;
}

_mysql_field_::_mysql_field_(int ntype, const Buffer& name, unsigned attr, const Buffer& type, const Buffer& size, const Buffer& default_, const Buffer& check)
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

_mysql_field_::_mysql_field_(const _mysql_field_& field)
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

_mysql_field_::~_mysql_field_()
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

Buffer _mysql_field_::Create()
{
    Buffer sql = "`" + Name + "` " + Type + Size + " ";
    if (Attr & NOT_NULL) {
        sql += "NOT NULL";
    }
    else {
        sql += "NULL";
    }
    //BLOB TEXT GEOMETRY JSON不能有默认值的
    if ((Attr & DEFAULT) && (Default.size() > 0)&&(Type != "BLOB") && (Type != "TEXT") && (Type != "GEOMETRY") && (Type != "JSON"))
    {
        sql += " DEFAULT \"" + Default + "\" ";
    }
    //UNIQUE PRIMARY_KEY 外面处理
    //CHECK mysql不支持
    if (Attr & AUTOINCREMENT) {
        sql += " AUTO_INCREMENT ";
    }
    return sql;
}

void _mysql_field_::LoadFromStr(const Buffer& str)
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
            printf("type=%d\n", nType);
            break;
    }
}

Buffer _mysql_field_::toEqualExp() const
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
            printf("type=%d\n", nType);
            break;
    }
    return sql;
}

Buffer _mysql_field_::toSqlStr() const
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
            printf("type=%d\n", nType);
            break;
    }
    return sql;
}

_mysql_field_::operator const Buffer() const
{
    return '`' + Name + '`';
}

Buffer _mysql_field_::Str2Hex(const Buffer& data) const
{
    const char* hex = "0123456789ABCDEF";
    std::stringstream ss;
    for (auto ch : data)
        ss << hex[(unsigned char)ch >> 4] << hex[(unsigned char)ch & 0xF];
    return ss.str();
}
