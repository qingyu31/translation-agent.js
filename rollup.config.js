export default [
    {
        input: 'src/index.js',
        output: {
            file: 'dist/index.esm.js',
            format: 'esm'
        }
    },
    {
        input: 'src/index.js',
        output: {
            file: 'dist/index.cjs.js',
            format: 'cjs'
        }
    },
    {
        input: 'src/index.js',
        output: {
            file: 'dist/index.umd.js',
            format: 'umd',
            name: 'translation-agent',
        }
    }
]
