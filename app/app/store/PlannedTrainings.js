var url = window.location.hostname;
var create = 'http://' + url + ':5001/new_planned_training';
var read = 'http://' + url + ':5001/get_planned_trainings';
var update = 'http://' + url + ':5001/update_planned_training';
var destroy = 'http://' + url + ':5001/delete_planned_training';
Ext.define('ClassificationApp.store.PlannedTrainings', {
    extend: 'Ext.data.Store',

    alias: 'store.plannedtrainings',

    model: 'ClassificationApp.model.PlannedTraining',
	proxy: {
        type: 'ajax',
		api: {
			create  : create,
			read    : read,
			update  : update,
			destroy : destroy
        },
        actionMethods: {
            create: 'POST',
            read: 'GET',
            update: 'POST',
            destroy: 'POST'
        },
        writer: {
            type: 'json',
            writeAllFields: true,
        },
        reader: {
            type: 'json',
            rootProperty: 'data'
        }
	},
});