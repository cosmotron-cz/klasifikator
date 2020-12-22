/*
 * This file launches the application by asking Ext JS to create
 * and launch() the Application class.
 */
Ext.application({
    extend: 'ClassificationApp.Application',

    name: 'ClassificationApp',

    requires: [
        // This will automatically load all classes in the ClassificationApp namespace
        // so that application classes do not need to require each other.
        'ClassificationApp.*'
    ],

    // The name of the initial view to create.
    mainView: 'ClassificationApp.view.main.Main'
});
