import js from '@eslint/js';
import ts from 'typescript-eslint';
export default ts.config({ignores:['dist/**','node_modules/**']},js.configs.recommended,...ts.configs.recommended,{files:['**/*.ts','**/*.tsx'],languageOptions:{globals:{window:'readonly',document:'readonly',fetch:'readonly',XMLHttpRequest:'readonly',AbortController:'readonly',URL:'readonly',console:'readonly',setTimeout:'readonly',clearTimeout:'readonly',HTMLAudioElement:'readonly',HTMLInputElement:'readonly',File:'readonly',Blob:'readonly',RequestInit:'readonly',process:'readonly'}}});
