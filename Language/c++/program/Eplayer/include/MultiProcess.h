//
// Created by buntu on 2023/7/24.
//

#ifndef EPLAYER_MULTIPROCESS_H
#define EPLAYER_MULTIPROCESS_H

class CFunctionBase;
template<typename _FUNCTION_,typename... _ARGS_>
class CFunction;
class CProcess;
int CreateClientServer(CProcess * proc);
int testMultiProcess();

#endif //EPLAYER_MULTIPROCESS_H
