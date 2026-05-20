import 'package:flutter/material.dart';

enum Status { active, inactive, pending }

mixin Loggable {
  void log(String message) {
    print('[${runtimeType}] $message');
  }
}

abstract class BaseWidget extends StatelessWidget {
  const BaseWidget({super.key});
}

class MyWidget extends BaseWidget with Loggable {
  final String title;

  const MyWidget({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(title)),
      body: const Center(child: Text('Hello Flutter')),
    );
  }
}

class Counter extends StatefulWidget {
  const Counter({super.key});

  @override
  State<Counter> createState() => _CounterState();
}

class _CounterState extends State<Counter> {
  int _count = 0;

  void _increment() {
    setState(() => _count++);
  }

  @override
  Widget build(BuildContext context) {
    return Text('$_count');
  }
}

extension StringExtension on String {
  String capitalize() {
    if (isEmpty) return this;
    return '${this[0].toUpperCase()}${substring(1)}';
  }
}

String formatCurrency(double amount) {
  return '\$${amount.toStringAsFixed(2)}';
}
