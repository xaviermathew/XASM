import operator
import time


class Clock(object):
    def __init__(self, ticks_per_second, cpu):
        self.i = 0
        self.connections = []
        self.ticks_per_second = ticks_per_second
        self.seconds_per_tick = 1.0 / ticks_per_second
        self.cpu = cpu

    def tick(self):
        self.i += 1
        for l in self.connections:
            l.tick()

    def add_connection(self, l):
        self.connections.append(l)

    def start(self):
        while True:
            print(x)
            self.tick()
            if self.cpu.control_bus.read_state() == ControlBus.EXITED:
                break
            else:
                time.sleep(self.seconds_per_tick)

    def __repr__(self):
        return '<%s: [%s] with [%s] connections>' % (self.__class__.__name__, self.i, len(self.connections))


class RAM(object):
    def __init__(self, num_bytes):
        self.num_bytes = num_bytes
        self.memory = [None] * num_bytes

    def __repr__(self):
        return '<%s: total:[%s] used:[%s]>' % (self.__class__.__name__, self.num_bytes, len(list(filter(bool, self.memory))))


class Latch(object):
    def __init__(self):
        self.value = None

    def read(self):
        return self.value

    def write(self, value):
        self.value = value

    def is_set(self):
        return self.value is not None

    def reset(self):
        self.value = None

    def __repr__(self):
        return '<%s: [%s]>' % (self.__class__.__name__, self.value)


class Register(Latch):
    pass


class ProgramCounter(Register):
    def __init__(self):
        self.write(0)

    def reset(self):
        self.write(0)

class BuiltinRegisters(object):
    components = [
        'pc', 'mar', 'mdr', 'ax', 'zero', 'sp', 'retp',
        'cir', 'op1', 'op2',
    ]

    def __init__(self):
        self.pc = ProgramCounter()
        self.mar = Register()
        self.mdr = Register()
        self.ax = Register()
        self.zero = Register()
        self.sp = Register()
        self.retp = Register()

        self.cir = Register()
        self.op1 = Register()
        self.op2 = Register()

    def reset(self):
        for reg in self.components:
            getattr(self, reg).reset()

    def __repr__(self):
        return '<%s\n%s\n>' % (self.__class__.__name__, '\n'.join('%s:%s' % (reg, getattr(self, reg)) for reg in self.components))


class GeneralPurposeRegisters(object):
    def __init__(self, num_registers):
        self.num_registers = num_registers
        for i in range(num_registers):
            setattr(self, 'r%s' % i, Register())

    def reset(self):
        for i in range(self.num_registers):
            r = getattr(self, 'r%s' % i)
            r.reset()

    def __repr__(self):
        reg_repr = '\n'.join('r%s:%s' % (i, repr(getattr(self, 'r%s' % i))) for i in range(self.num_registers))
        return '<%s\n%s\n>' % (self.__class__.__name__, reg_repr)


class Step(object):
    def __init__(self, fetcher):
        self.fetcher = fetcher

    def tick(self):
        pass


class Fetcher(object):
    def __init__(self, cpu):
        self.cpu = cpu

    def tick(self):
        cpu = self.cpu
        executor = cpu.executor
        bi_registers = cpu.builtin_registers
        control_bus = cpu.control_bus
        address_bus = cpu.address_bus
        data_bus = cpu.data_bus
        ram = cpu.ram
        state = control_bus.read_state()

        op_fetch_states = {ControlBus.COPY_FROM_RAM, ControlBus.COPY_OP_FROM_RAM, ControlBus.COPY_OP1_FROM_RAM, ControlBus.COPY_OP2_FROM_RAM}
        ram_fetch_states = {ControlBus.COPY_FROM_RAM}.union(op_fetch_states)
        op_to_cpu_states = {ControlBus.COPY_OP_TO_CPU, ControlBus.COPY_OP1_TO_CPU, ControlBus.COPY_OP2_TO_CPU}
        to_cpu_states = {ControlBus.COPY_DATA_TO_CPU}.union(op_to_cpu_states)

        # ram to cpu
        # import pdb; pdb.set_trace()
        if state == ControlBus.IDLE and bi_registers.mar.is_set():
            control_bus.write_next_state(ControlBus.COPY_FROM_RAM)
        elif state in ram_fetch_states and bi_registers.mar.is_set() and not address_bus.is_set() and not data_bus.is_set():
            address_bus.write(bi_registers.mar.read())
            bi_registers.mar.reset()
        elif state in ram_fetch_states and not bi_registers.mar.is_set() and address_bus.is_set() and not data_bus.is_set():
            address = address_bus.read()
            address_bus.reset()
            value = ram.memory[address]
            if value is None:
                control_bus.write_next_state(ControlBus.EXITED)
            else:
                data_bus.write(value)
                if state in op_fetch_states:
                    if state == ControlBus.COPY_OP_FROM_RAM:
                        control_bus.write_next_state(ControlBus.COPY_OP_TO_CPU)
                    elif state == ControlBus.COPY_OP1_FROM_RAM:
                        control_bus.write_next_state(ControlBus.COPY_OP1_TO_CPU)
                    elif state == ControlBus.COPY_OP2_FROM_RAM:
                        control_bus.write_next_state(ControlBus.COPY_OP2_TO_CPU)
                else:
                    control_bus.write_next_state(ControlBus.COPY_DATA_TO_CPU)
        elif state in to_cpu_states and data_bus.is_set() and not bi_registers.mdr.is_set():
            value = data_bus.read()
            data_bus.reset()
            bi_registers.mdr.write(value)
        elif state in to_cpu_states and not data_bus.is_set() and bi_registers.mdr.is_set():
            value = bi_registers.mdr.read()
            if state in op_to_cpu_states:
                bi_registers.mdr.reset()
                if state == ControlBus.COPY_OP_TO_CPU:
                    bi_registers.cir.write(value)
                    bi_registers.mar.write(bi_registers.pc.read() + 1)
                    control_bus.write_next_state(ControlBus.COPY_OP1_FROM_RAM)
                elif state == ControlBus.COPY_OP1_TO_CPU:
                    bi_registers.op1.write(value)
                    bi_registers.mar.write(bi_registers.pc.read() + 2)
                    control_bus.write_next_state(ControlBus.COPY_OP2_FROM_RAM)
                elif state == ControlBus.COPY_OP2_TO_CPU:
                    bi_registers.op2.write(value)
                    bi_registers.pc.write(bi_registers.pc.read() + 3)
                    control_bus.write_next_state(ControlBus.START_EXECUTOR)
            else:
                control_bus.write_next_state(ControlBus.IDLE)

        # cpu to ram
        elif state == ControlBus.IDLE and bi_registers.mar.is_set() and bi_registers.mdr.is_set():
            control_bus.write_next_state(ControlBus.COPY_DATA_FROM_CPU)
        elif state == ControlBus.COPY_DATA_FROM_CPU and bi_registers.mar.is_set() and bi_registers.mdr.is_set() and not address_bus.is_set() and not data_bus.is_set():
            value = bi_registers.mdr.read()
            data_bus.write(value)
            address = bi_registers.mar.read()
            bi_registers.mar.reset()
            address_bus.write(address)
            control_bus.write_next_state(ControlBus.COPY_DATA_TO_RAM)
        elif state == ControlBus.COPY_DATA_TO_RAM and address_bus.is_set() and data_bus.is_set():
            address = address_bus.read()
            address_bus.reset()
            value = data_bus.read()
            data_bus.reset()
            ram.memory[address] = value
            control_bus.write_next_state(ControlBus.IDLE)

        # instruction cycle
        elif state == ControlBus.IDLE and bi_registers.pc.is_set() and not bi_registers.mar.is_set():
            address = bi_registers.pc.read()
            bi_registers.mar.write(address)
            control_bus.write_next_state(ControlBus.COPY_OP_FROM_RAM)

        # execute instructions
        elif state == ControlBus.START_EXECUTOR:
            executor.tick()


class MoveValueToRegister(Step):
    def __init__(self, executor):
        self.executor = executor

    def tick(self):
        cpu = self.executor.cpu
        bi_registers = cpu.builtin_registers

        value = bi_registers.op1.read()
        register_num = bi_registers.op2.read()
        getattr(cpu.gp_registers, register_num).write(value)

        cpu.control_bus.write_next_state(ControlBus.IDLE)


class MoveRegisterToRegister(Step):
    def __init__(self, executor):
        self.executor = executor

    def tick(self):
        cpu = self.executor.cpu
        bi_registers = cpu.builtin_registers
        gp_registers = cpu.gp_registers

        src = bi_registers.op1.read()
        dest = bi_registers.op2.read()
        if src.startswith('r'):
            value = getattr(gp_registers, src).read()
        else:
            value = getattr(bi_registers, src).read()

        if dest.startswith('r'):
            getattr(gp_registers, dest).write(value)
        else:
            getattr(bi_registers, dest).write(value)

        cpu.control_bus.write_next_state(ControlBus.IDLE)


class MovePointerToRegister(Step):
    def __init__(self, executor):
        self.executor = executor

    def tick(self):
        cpu = self.executor.cpu
        bi_registers = cpu.builtin_registers

        raise NotImplementedError
        src = None
        dest = bi_registers.op2.read()
        value = None
        getattr(cpu.gp_registers, dest).write(value)

        cpu.control_bus.write_next_state(ControlBus.IDLE)


class OperatorBase(Step):
    operator = None
    destination = None

    def __init__(self, executor):
        self.executor = executor

    def tick(self):
        cpu = self.executor.cpu
        bi_registers = cpu.builtin_registers
        gp_registers = cpu.gp_registers

        lhs = bi_registers.op1.read()
        rhs = bi_registers.op2.read()
        lhs_reg = getattr(gp_registers, lhs)
        rhs_reg = getattr(gp_registers, rhs)

        value = self.operator(lhs_reg.read(), rhs_reg.read())
        dest = getattr(bi_registers, self.destination)
        dest.write(value)

        cpu.control_bus.write_next_state(ControlBus.IDLE)


class ArithmeticBase(OperatorBase):
    destination = 'ax'


class Add(ArithmeticBase):
    operator = operator.add


class Subtract(ArithmeticBase):
    operator = operator.sub


class Multiply(ArithmeticBase):
    operator = operator.mul


class Divide(ArithmeticBase):
    operator = operator.floordiv


class LogicBase(OperatorBase):
    destination = 'zero'


class And(LogicBase):
    operator = operator.and_


class Or(LogicBase):
    operator = operator.or_


class Equals(LogicBase):
    operator = operator.eq


class NotEquals(LogicBase):
    operator = operator.ne


class LessThan(LogicBase):
    operator = operator.lt


class GreaterThan(LogicBase):
    operator = operator.gt


class LessThanOrEquals(LogicBase):
    operator = operator.le


class GreaterThanOrEquals(LogicBase):
    operator = operator.ge


class Jump(Step):
    def __init__(self, executor):
        self.executor = executor

    def tick(self):
        cpu = self.executor.cpu
        bi_registers = cpu.builtin_registers
        gp_registers = cpu.gp_registers

        address = bi_registers.op1.read()
        bi_registers.pc.write(address)

        cpu.control_bus.write_next_state(ControlBus.IDLE)


class JumpEquals(Step):
    def __init__(self, executor):
        self.executor = executor

    def tick(self):
        cpu = self.executor.cpu
        bi_registers = cpu.builtin_registers

        check = bi_registers.zero.read()
        if check:
            address = bi_registers.op1.read()
            bi_registers.pc.write(address)

        cpu.control_bus.write_next_state(ControlBus.IDLE)


class Not(Step):
    pass


class Call(Step):
    pass


class Return(Step):
    pass


class Executor(object):
    MOV_VAL = 'MOV_VAL'
    MOV_REG = 'MOV_REG'
    MOV_PTR = 'MOV_PTR'
    ADD = 'ADD'
    SUB = 'SUB'
    MUL = 'MUL'
    DIV = 'DIV'
    AND = 'AND'
    OR = 'OR'
    EQ = 'EQ'
    NEQ = 'NEQ'
    LT = 'LT'
    GT = 'GT'
    LTE = 'LTE'
    GTE = 'GTE'

    JMP = 'JMP'
    JEQ = 'JEQ'
    NOT = 'NOT'

    CALL = 'CALL'
    RET = 'RET'

    instruction_set = {
        MOV_VAL: (2, MoveValueToRegister),
        MOV_REG: (2, MoveRegisterToRegister),
        MOV_PTR: (2, MovePointerToRegister),
        ADD: (2, Add),
        SUB: (2, Subtract),
        MUL: (2, Multiply),
        DIV: (2, Divide),
        AND: (2, And),
        OR: (2,  Or),
        EQ: (2, Equals),
        NEQ: (2, NotEquals),
        LT: (2, LessThan),
        GT: (2, GreaterThan),
        LTE: (2, LessThanOrEquals),
        GTE: (2, GreaterThanOrEquals),

        JMP: (1, Jump),
        JEQ: (1, JumpEquals),
        NOT: (1, Not),
        CALL: (1, Call),

        RET: (0, Return),
    }

    def __init__(self, cpu):
        self.cpu = cpu
        self.instructions = {
            op: op_class(self)
            for op, (num_operands, op_class) in self.instruction_set.items()
        }

    def tick(self):
        cpu = self.cpu
        state = cpu.control_bus.read_state()
        if state != ControlBus.START_EXECUTOR:
            return

        bi_registers = cpu.builtin_registers
        cir = bi_registers.cir
        self.instructions[cir.read()].tick()


class AddressBus(Latch):
    pass


class DataBus(Latch):
    pass


class ControlBus(object):
    IDLE = 'IDLE'
    EXITED = 'EXITED'
    UNSET = 'UNSET'

    COPY_FROM_RAM = 'COPY_FROM_RAM'
    COPY_DATA_TO_RAM = 'COPY_DATA_TO_RAM'
    COPY_DATA_FROM_CPU = 'COPY_DATA_FROM_CPU'
    COPY_DATA_TO_CPU = 'COPY_DATA_TO_CPU'

    COPY_OP_FROM_RAM = 'COPY_OP_FROM_RAM'
    COPY_OP1_FROM_RAM = 'COPY_OP1_FROM_RAM'
    COPY_OP2_FROM_RAM = 'COPY_OP2_FROM_RAM'

    COPY_OP_TO_CPU = 'COPY_OP_TO_CPU'
    COPY_OP1_TO_CPU = 'COPY_OP1_TO_CPU'
    COPY_OP2_TO_CPU = 'COPY_OP2_TO_CPU'

    START_EXECUTOR = 'START_EXECUTOR'

    def __init__(self, cpu):
        self.cpu = cpu
        self.state = self.IDLE
        self.next_state = self.UNSET

    def read_state(self):
        return self.state

    def write_next_state(self, state):
        self.next_state = state

    def reset(self):
        self.state = self.IDLE
        self.next_state = self.IDLE

    def tick(self):
        if self.next_state != self.UNSET:
            self.state = self.next_state
            self.next_state = self.UNSET

    def __repr__(self):
        return '<%s: now:[%s] next:[%s]>' % (self.__class__.__name__, self.state, self.next_state)


class CPU(object):
    components = [
        'fetcher', 'executor', 'address_bus', 'data_bus',
        'ram', 'builtin_registers', 'gp_registers',
        'clock', 'control_bus'
    ]

    def __init__(self, num_bytes, num_registers, ticks_per_second):
        self.fetcher = Fetcher(self)
        self.executor = Executor(self)
        self.address_bus = AddressBus()
        self.data_bus = DataBus()
        self.control_bus = ControlBus(self)
        self.ram = RAM(num_bytes)
        self.builtin_registers = BuiltinRegisters()
        self.gp_registers = GeneralPurposeRegisters(num_registers)

        c = Clock(ticks_per_second, self)
        c.add_connection(self)
        self.clock = c

    def reset(self):
        for c in self.components:
            c = getattr(self, c)
            if hasattr(c, 'reset'):
                c.reset()

    def tick(self):
        self.control_bus.tick()
        self.fetcher.tick()

    def __repr__(self):
        components = [
            self.clock,
            self.control_bus, self.data_bus, self.address_bus,
            self.builtin_registers, self.gp_registers,
            self.ram,
        ]
        return '<%s\n%s\n>' % (self.__class__.__name__, '\n'.join('%s' % c for c in components))


rom = [
int(t) if t.isdigit() else t
for line in
"""
MOV_VAL 10 r0
MOV_VAL 20 r1
ADD r0 r1
MOV_REG ax r1
JMP 6 0
""".splitlines()
for t in line.split() if t]
x = CPU(num_bytes=len(rom) + 20, num_registers=2, ticks_per_second=100)
for i, word in enumerate(rom):
    x.ram.memory[i] = word
x.clock.start()
