const vue = require('eslint-plugin-vue');

module.exports = [
  {
    ignores: ['dist/**', 'node_modules/**'],
  },
  ...vue.configs['flat/essential'],
  {
    rules: {
      // The codebase passes reactive objects as props intentionally (e.g. chat state).
      'vue/no-mutating-props': 'off',
      // Many Vue SFCs in this repo are intentionally single-word.
      'vue/multi-word-component-names': 'off',
      'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    },
  },
];
