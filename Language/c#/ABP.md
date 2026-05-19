# ABP

ABP（ASP.NET Boilerplate）是**C#/.NET 生态最主流的企业级开发框架**

ABP对应的是Spring全家桶

# HTTP 服务

在http服务框架中，用`C# ABP` 对比 `Java SpringBoot`

| Java SpringBoot   | C# ABP 框架                                 | 作用完全一致吗？                 |
| ----------------- | ------------------------------------------- | -------------------------------- |
| **Controller 层** | **Controller / Application Service 接口层** | ✅ 完全一致（接收 HTTP 请求）     |
| **Service 层**    | **Domain Service / Application Service**    | ✅ 完全一致（业务逻辑）           |
| **Entity 层**     | **Entity 实体类**                           | ✅ 完全一致（数据库映射对象）     |
| **Model 层**      | **DTO / ViewModel**                         | ✅ 完全一致（数据传输、接口返回） |
| **DAO层**         | **IRepository**                             | ✅ 完全一致（数据库操作）         |

**Controller**

```java
@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private UserService userService;

    @GetMapping("/{id}")
    public UserDto getUserById(@PathVariable Long id) {
        return userService.getUserById(id);
    }
}
```

```c#
[Route("api/users")]
public class UserController : AbpController
{
    private readonly IUserAppService _userAppService;

    // 构造函数注入（ABP默认依赖注入）
    public UserController(IUserAppService userAppService)
    {
        _userAppService = userAppService;
    }

    [HttpGet("{id}")]
    public async Task<UserDto> GetAsync(Guid id)
    {
        return await _userAppService.GetAsync(id);
    }
}
```

**Service**

```java
@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    public UserDto getUserById(Long id) {
        User user = userRepository.findById(id).orElse(null);
        return new UserDto(user);
    }
}
```

```c#
public class UserAppService : ApplicationService, IUserAppService
{
    private readonly IRepository<User, Guid> _userRepository;

    public UserAppService(IRepository<User, Guid> userRepository)
    {
        _userRepository = userRepository;
    }

    public async Task<UserDto> GetAsync(Guid id)
    {
        User user = await _userRepository.GetAsync(id);
        return ObjectMapper.Map<User, UserDto>(user);
    }
}
```

**Entity**

```java
@Entity
@Table(name = "users")
public class User {
    @Id
    private Long id;
    private String name;
    private int age;
    // getter/setter
}
```

```c#
public class User : Entity<Guid>
{
    public string Name { get; set; }
    public int Age { get; set; }
}
```

**DTO**

```c#
public class UserDto {
    private Long id;
    private String name;
    private int age;
}
```

```c#
public class UserDto : EntityDto<Guid>
{
    public string Name { get; set; }
    public int Age { get; set; }
}
```

